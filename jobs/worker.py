from __future__ import annotations
import argparse, logging, os, sys, time

# Import your pipeline pieces
from data.universe import rebuild_universe
from data.ingest import ingest_bars_for_universe
from run_pipeline import main as run_eod_pipeline

def main():
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - worker - %(levelname)s - %(message)s"
    )
    log = logging.getLogger("worker")

    parser = argparse.ArgumentParser(description="Blank Capital worker")
    parser.add_argument(
        "task",
        nargs="?",
        choices=["idle", "universe", "ingest", "pipeline", "celery"],
        default=os.getenv("WORKER_TASK", "idle"),
        help="Task to run (default: from WORKER_TASK env var, or 'idle')"
    )
    parser.add_argument("--days", type=int, default=int(os.getenv("DAYS", "7")))
    parser.add_argument("--celery-args", type=str, default="", help="Additional Celery worker arguments")
    args = parser.parse_args()

    log.info("Starting worker task=%s", args.task)
    try:
        if args.task == "idle":
            # Keep the worker alive so you can exec ad-hoc commands in the shell
            while True:
                time.sleep(3600)
        elif args.task == "universe":
            rebuild_universe()
        elif args.task == "ingest":
            ingest_bars_for_universe(days=args.days)
        elif args.task == "pipeline":
            ok = run_eod_pipeline(sync_broker=os.getenv("SYNC_TO_BROKER", "false").lower() == "true")
            if not ok:
                sys.exit(1)
        elif args.task == "celery":
            # Start Celery worker
            try:
                from tasks.celery_app import celery_app
                # Default Celery worker args
                celery_args = [
                    'worker',
                    '--loglevel=info',
                    '--concurrency=2',
                ]
                if args.celery_args:
                    celery_args.extend(args.celery_args.split())
                
                log.info("Starting Celery worker with args: %s", celery_args)
                celery_app.start(celery_args)
            except ImportError:
                log.error("Celery not available. Install celery[redis] to use this mode.")
                sys.exit(1)
    except Exception:
        log.exception("Worker task failed")
        sys.exit(1)

    log.info("Worker task complete")
    sys.exit(0)

if __name__ == "__main__":
    main()
