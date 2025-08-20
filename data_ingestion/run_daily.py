# Cron/worker entrypoint to run ingestion once.
import os
from datetime import datetime, timedelta
from .main import SYMBOLS, fetch_and_store


if __name__ == "__main__":
    days = int(os.environ.get("BACKFILL_DAYS", str(365 * 2)))  # default 2 years
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    try:
        written = fetch_and_store(SYMBOLS, start, end)
        print(
            f"[run_daily] Wrote {written} rows from {start} to {end} (days={days})"
        )
    except Exception as exc:
        print(f"[run_daily] Failed to write data: {exc}")
