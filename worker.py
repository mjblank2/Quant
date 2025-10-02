"""
CLI entrypoint for the Blank Capital worker.

This module exposes a ``main()`` function and can be executed directly.
Depending on the command‑line arguments (or the ``WORKER_TASK``
environment variable), the worker will idle, rebuild the universe,
ingest market bars, run the end‑of‑day pipeline, or start a Celery
worker.  Logging and error handling are configured centrally.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from data.universe import rebuild_universe
from data.ingest import ingest_bars_for_universe


try:
    # Import the end‑of‑day pipeline if available; otherwise provide a
    # stub that returns True.  Import errors here should not crash the
    # worker itself.
    from run_pipeline import main as run_eod_pipeline  # type: ignore
except Exception:
    def run_eod_pipeline(*_args, **_kwargs):  # type: ignore[return-type]
        logging.getLogger("worker").warning(
            "run_eod_pipeline not available in this build."
        )
        return True


def main() -> None:
    """Entry point for the quant worker.

    Determines which task to run based on the command‑line arguments or the
    ``WORKER_TASK`` environment variable.  Supports idling, universe
    rebuilds, bar ingestion, EOD pipeline execution, and Celery worker
    startup.  Any unhandled exception will be logged and cause the
    process to exit with a non‑zero status.
    """

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - worker - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("worker")

    parser = argparse.ArgumentParser(description="Blank Capital worker")
    parser.add_argument(
        "task",
        nargs="?",
        choices=["idle", "universe", "ingest", "pipeline", "celery"],
        default=os.getenv("WORKER_TASK", "idle"),
        help="Task to run (default: from WORKER_TASK env var, or 'idle')",
    )
    parser.add_argument("--days", type=int, default=int(os.getenv("DAYS", "7")))
    parser.add_argument(
        "--celery-args",
        type=str,
        default="",
        help="Additional Celery worker arguments",
    )
    args = parser.parse_args()

    log.info("Starting worker task=%s", args.task)
    try:
        if args.task == "idle":
            # Idle loop keeps the process alive without performing any work.
            while True:
                time.sleep(3600)
        elif args.task == "universe":
            rebuild_universe()
        elif args.task == "ingest":
            ingest_bars_for_universe(days=args.days)
        elif args.task == "pipeline":
            # The EOD pipeline may return False/None on failure; exit with 1.
            ok = run_eod_pipeline(
                sync_broker=os.getenv("SYNC_TO_BROKER", "false").lower() == "true"
            )
            if not ok:
                sys.exit(1)
        elif args.task == "celery":
            # Launch a Celery worker if the tasks package is installed.
            try:
                from tasks.celery_app import celery_app  # type: ignore

                celery_args = ["worker", "--loglevel=info", "--concurrency=2"]
                if args.celery_args:
                    celery_args.extend(args.celery_args.split())
                log.info("Starting Celery worker with args: %s", celery_args)
                celery_app.start(celery_args)
            except ImportError:
                log.error(
                    "Celery not available. Install celery[redis] to use this mode."
                )
                sys.exit(1)
    except Exception:
        log.exception("Worker task failed")
        sys.exit(1)

    log.info("Worker task complete")
    sys.exit(0)


if __name__ == "__main__":
    main()

