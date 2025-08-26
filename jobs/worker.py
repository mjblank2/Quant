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
        choices=["idle", "universe", "ingest", "pipeline"],
        default=os.getenv("WORKER_TASK", "idle"),
        help="Task to run (default: from WORKER_TASK env var, or 'idle')"
    )
    parser.add_argument("--days", type=int, default=int(os.getenv("DAYS", "7")))
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
    except Exception:
        log.exception("Worker task failed")
        sys.exit(1)

    log.info("Worker task complete")
    sys.exit(0)

if __name__ == "__main__":
    main()
