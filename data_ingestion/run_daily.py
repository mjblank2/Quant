# Cron/worker entrypoint to run ingestion once.
import os
from datetime import datetime, timedelta
from .main import fetch_and_store, SYMBOLS

if __name__ == "__main__":
    days = int(os.environ.get("BACKFILL_DAYS", str(365*2)))  # default 2 years
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    written = fetch_and_store(SYMBOLS, start, end)
    print(f"[run_daily] Wrote {written} rows from {start} to {end} (days={days})")
