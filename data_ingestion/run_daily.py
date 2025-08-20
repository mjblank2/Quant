"""
Entry point for running the ingestion job as a one‑off.

This module reads the ``BACKFILL_DAYS`` environment variable (defaulting
to two years if unspecified) and invokes ``fetch_and_store`` from
``data_ingestion.main``.  It is intended to be used as the command
for worker and cron services in Render.
"""

import os
from datetime import datetime, timedelta

from .main import fetch_and_store, SYMBOLS


if __name__ == "__main__":
    # Determine lookback period.  Use BACKFILL_DAYS env var if set, otherwise
    # default to two years.  A string default is accepted to mirror typical
    # configuration patterns on platforms like Render.
    try:
        days = int(os.environ.get("BACKFILL_DAYS", str(365 * 2)))
    except ValueError:
        days = 365 * 2
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    written = fetch_and_store(SYMBOLS, start_time, end_time)
    print(f"[run_daily] Wrote {written} rows from {start_time} to {end_time} (days={days})")
