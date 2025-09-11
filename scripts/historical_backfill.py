from __future__ import annotations
import os, sys, time, traceback, math, datetime as dt
from sqlalchemy import create_engine, text
from typing import Optional

# You may already have this; adjust import if path differs
from data.ingest import ingest_bars_for_universe

TARGET_LOOKBACK_DAYS = int(os.getenv("BACKFILL_TARGET_DAYS", "420"))
MAX_CUMULATIVE_CAP   = int(os.getenv("BACKFILL_MAX_CAP", str(TARGET_LOOKBACK_DAYS)))
MIN_STEP             = int(os.getenv("BACKFILL_MIN_STEP", "15"))
MAX_STEP             = int(os.getenv("BACKFILL_MAX_STEP", "90"))
SLOW_SEC             = int(os.getenv("BACKFILL_SLOW_THRESHOLD", "240"))

STATE_TABLE = "ingestion_progress"

def _engine():
    db=os.getenv("DATABASE_URL")
    if not db:
        raise RuntimeError("DATABASE_URL missing")
    if db.startswith("postgres://"):
        db=db.replace("postgres://","postgresql+psycopg://",1)
    return create_engine(db)

def ensure_state_table():
    e=_engine()
    with e.begin() as c:
        c.execute(text(f"""
          CREATE TABLE IF NOT EXISTS {STATE_TABLE} (
            id SERIAL PRIMARY KEY,
            run_ts TIMESTAMP DEFAULT NOW(),
            cumulative_days INT NOT NULL,
            duration_sec DOUBLE PRECISION,
            success BOOLEAN,
            error TEXT
          )
        """))

def latest_cumulative() -> int:
    e=_engine()
    with e.connect() as c:
        val = c.execute(text(f"SELECT cumulative_days FROM {STATE_TABLE} WHERE success IS TRUE ORDER BY id DESC LIMIT 1")).scalar()
        return int(val) if val else 0

def record(cum: int, dur: Optional[float], success: bool, error: Optional[str]):
    e=_engine()
    with e.begin() as c:
        c.execute(text(f"""
          INSERT INTO {STATE_TABLE}(cumulative_days, duration_sec, success, error)
          VALUES(:c,:d,:s,:e)
        """), {"c": cum, "d": dur, "s": success, "e": error})

def main():
    ensure_state_table()
    current = latest_cumulative()
    target  = min(TARGET_LOOKBACK_DAYS, MAX_CUMULATIVE_CAP)
    if current >= target:
        print(f"Already at or beyond target ({current} >= {target}), nothing to do.")
        return 0

    print(f"Resuming backfill from {current} -> {target} days.")
    while current < target:
        remaining = target - current
        # Step heuristic: proportional but bounded
        step = min(MAX_STEP, max(MIN_STEP, remaining // 3))
        next_cumulative = min(target, current + step)
        print(f"Ingest attempt: cumulative {next_cumulative} days (step={next_cumulative-current})")
        t0 = time.time()
        try:
            ingest_bars_for_universe(days=next_cumulative)
            dur = time.time() - t0
            record(next_cumulative, dur, True, None)
            print(f"Success cumulative={next_cumulative} dur={dur:.1f}s")
            # Adjust step adaptively: if very fast, allow larger leaps; if slow, smaller
            if dur > SLOW_SEC:
                # Lower MAX_STEP temporarily
                global MAX_STEP
                MAX_STEP = max(MIN_STEP, MAX_STEP // 2)
                print(f"Slow chunk; reducing MAX_STEP to {MAX_STEP}")
            current = next_cumulative
        except Exception as e:
            dur = time.time() - t0
            err = f"{type(e).__name__}: {e}"
            record(next_cumulative, dur, False, err)
            traceback.print_exc()
            # Back off: shrink MAX_STEP drastically
            MAX_STEP = max(MIN_STEP, MAX_STEP // 2)
            print(f"Error; backing off. New MAX_STEP={MAX_STEP}. Sleeping 15s.")
            time.sleep(15)
    print("Historical backfill complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
