from __future__ import annotations
import argparse, logging, os, sys, time
from data.universe import _list_alpaca_assets as rebuild_universe  # placeholder
from data.ingest import _bars_from_alpaca_batch as ingest_bars_for_universe  # placeholder
from models.train_predict import train_and_predict_all_models as run_eod_pipeline  # placeholder

def main():
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s - worker - %(levelname)s - %(message)s")
    log = logging.getLogger("worker")

    parser = argparse.ArgumentParser(description="Blank Capital worker")
    parser.add_argument("task", nargs="?", choices=["idle","universe","ingest","pipeline","celery"], default=os.getenv("WORKER_TASK","idle"))
    parser.add_argument("--days", type=int, default=int(os.getenv("DAYS","7")))
    args = parser.parse_args()

    log.info("Starting worker task=%s", args.task)
    try:
        if args.task == "idle":
            while True:
                time.sleep(3600)
        elif args.task == "universe":
            rebuild_universe()
        elif args.task == "ingest":
            ingest_bars_for_universe([], None, None)  # placeholders
        elif args.task == "pipeline":
            ok = run_eod_pipeline()
            if not ok:
                sys.exit(1)
        elif args.task == "celery":
            try:
                from tasks.celery_app import celery_app
                celery_args = [
                    'worker',
                    '--loglevel=info',
                    '--concurrency=2',
                ]
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
