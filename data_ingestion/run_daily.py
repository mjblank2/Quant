# data_ingestion/run_daily.py
import os
from datetime import datetime, timedelta
from .core import fetch_and_store, SYMBOLS

if __name__ == "__main__":
    try:
        days = int(os.environ.get("BACKFILL_DAYS", str(365 * 2)))
    except ValueError:
        days = 365 * 2
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    written = fetch_and_store(SYMBOLS, start_time, end_time)
    print(f"[run_daily] Wrote {written} rows from {start_time} to {end_time} (days={days})")
