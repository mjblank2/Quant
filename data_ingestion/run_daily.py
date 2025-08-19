# Cron/worker entrypoint to run ingestion once.
from datetime import datetime, timedelta
from .main import fetch_and_store, SYMBOLS

if __name__ == "__main__":
    end = datetime.utcnow()
    start = end - timedelta(days=365*2)
    written = fetch_and_store(SYMBOLS, start, end)
    print(f"[run_daily] Wrote {written} rows from {start} to {end}")
