from __future__ import annotations
import os
import sys
import time
import traceback
from typing import Optional

from sqlalchemy import create_engine, text

# Import your existing ingest function
from data.ingest import ingest_bars_for_universe

# ---------------- Configuration (override via env) ----------------
TARGET_LOOKBACK_DAYS = int(os.getenv("BACKFILL_TARGET_DAYS", "420"))
MIN_STEP             = int(os.getenv("BACKFILL_MIN_STEP", "15"))   # smallest incremental cumulative increase
MAX_STEP_DEFAULT     = int(os.getenv("BACKFILL_MAX_STEP", "90"))   # upper bound on per-iteration cumulative delta
SLOW_SEC             = int(os.getenv("BACKFILL_SLOW_THRESHOLD", "240"))  # If a chunk slower than this, shrink step
STATE_TABLE          = os.getenv("BACKFILL_STATE_TABLE", "ingestion_progress")
SLEEP_BETWEEN_SEC    = int(os.getenv("BACKFILL_SLEEP_BETWEEN_SEC", "3"))
SLEEP_ON_ERROR_SEC   = int(os.getenv("BACKFILL_SLEEP_ON_ERROR_SEC", "15"))

# ------------------------------------------------------------------

def _engine():
    db = os.getenv("DATABASE_URL")
    if not db:
        raise RuntimeError("DATABASE_URL missing")
    if db.startswith("postgres://"):
        db = db.replace("postgres://", "postgresql+psycopg://", 1)
    return create_engine(db)


def ensure_state_table():
    e = _engine()
    with e.begin() as c:
        c.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {STATE_TABLE} (
                id SERIAL PRIMARY KEY,
                run_ts TIMESTAMP DEFAULT NOW(),
                cumulative_days INT NOT NULL,
                step_days INT NOT NULL,
                duration_sec DOUBLE PRECISION,
                success BOOLEAN NOT NULL,
                error TEXT
            )
        """))


def latest_cumulative() -> int:
    e = _engine()
    with e.connect() as c:
        val = c.execute(
            text(f"SELECT cumulative_days FROM {STATE_TABLE} WHERE success IS TRUE ORDER BY id DESC LIMIT 1")
        ).scalar()
        return int(val) if val else 0


def record(cumulative: int, step: int, dur: Optional[float], success: bool, error: Optional[str]):
    e = _engine()
    with e.begin() as c:
        c.execute(text(f"""
            INSERT INTO {STATE_TABLE}(cumulative_days, step_days, duration_sec, success, error)
            VALUES (:cum, :step, :dur, :succ, :err)
        """),
        {"cum": cumulative, "step": step, "dur": dur, "succ": success, "err": error})


def _pick_step(remaining: int, last_duration: Optional[float], current_max_step: int) -> int:
    """
    Heuristic to decide next cumulative step size.
    - If last_duration is None (first run), pick moderate chunk.
    - If last chunk was slow, shrink allowable step.
    - Otherwise scale with remaining distance, bounded by current_max_step.
    """
    if remaining <= 0:
        return 0

    if last_duration is None:
        # First iteration: try about a third of remaining but bounded
        base = max(MIN_STEP, remaining // 3)
        return min(current_max_step, base)

    # If last chunk slow, reduce aggressiveness
    if last_duration > SLOW_SEC:
        new_cap = max(MIN_STEP, current_max_step // 2)
        # Keep step on the small side
        step = max(MIN_STEP, min(new_cap, remaining // 4 or MIN_STEP))
        return step

    # Otherwise we can be more aggressive but still bounded
    step = max(MIN_STEP, remaining // 3)
    return min(current_max_step, step)


def main() -> int:
    print(f"[historical_backfill] Starting with target={TARGET_LOOKBACK_DAYS} days")
    ensure_state_table()

    current = latest_cumulative()
    print(f"[historical_backfill] Resume detected: current cumulative={current}")

    if current >= TARGET_LOOKBACK_DAYS:
        print("[historical_backfill] Already at or beyond target; nothing to do.")
        return 0

    last_duration: Optional[float] = None
    max_step_cap = MAX_STEP_DEFAULT

    while current < TARGET_LOOKBACK_DAYS:
        remaining = TARGET_LOOKBACK_DAYS - current
        step = _pick_step(remaining, last_duration, max_step_cap)
        if step <= 0:
            print("[historical_backfill] Step calculated as 0; ending loop.")
            break

        next_cumulative = current + step
        if next_cumulative > TARGET_LOOKBACK_DAYS:
            next_cumulative = TARGET_LOOKBACK_DAYS
            step = next_cumulative - current

        print(f"[historical_backfill] Ingest attempt: new cumulative={next_cumulative} (step={step}, remaining after={TARGET_LOOKBACK_DAYS - next_cumulative})")
        t0 = time.time()
        error_msg = None
        success = True
        try:
            # NOTE: ingest function interprets 'days' as a total lookback horizon. Safe to re-call with larger values.
            ingest_bars_for_universe(days=next_cumulative)
        except Exception as e:
            success = False
            error_msg = f"{type(e).__name__}: {e}"
            traceback.print_exc()

        dur = time.time() - t0
        record(next_cumulative if success else current + step, step, dur, success, error_msg)
        print(f"[historical_backfill] Result: success={success} duration={dur:.1f}s")

        if success:
            current = next_cumulative
            last_duration = dur
            # Adjust cap if extremely fast or slow
            if dur > SLOW_SEC:
                max_step_cap = max(MIN_STEP, max_step_cap // 2)
                print(f"[historical_backfill] Slow chunk; reducing max_step_cap to {max_step_cap}")
            elif dur < SLOW_SEC / 4 and max_step_cap < MAX_STEP_DEFAULT:
                # Mildly increase (bounded by default)
                max_step_cap = min(MAX_STEP_DEFAULT, max_step_cap + MIN_STEP)
        else:
            # On failure: backoff
            last_duration = dur
            max_step_cap = max(MIN_STEP, max_step_cap // 2)
            print(f"[historical_backfill] Failure; backing off max_step_cap -> {max_step_cap}. Sleeping {SLEEP_ON_ERROR_SEC}s.")
            time.sleep(SLEEP_ON_ERROR_SEC)

        if current < TARGET_LOOKBACK_DAYS and success:
            time.sleep(SLEEP_BETWEEN_SEC)

    print(f"[historical_backfill] Completed or reached target. Final cumulative={current}")
    return 0 if current >= TARGET_LOOKBACK_DAYS else 1


if __name__ == "__main__":
    raise SystemExit(main())
