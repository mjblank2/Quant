# Back-compat shim: legacy `python -m data_ingestion.run_daily` forwards to v17 entry points.
from __future__ import annotations
import argparse, logging, os
from run_pipeline import main as run_eod_pipeline
from data.ingest import ingest_bars_for_universe
from data.universe import rebuild_universe
from data.fundamentals import fetch_fundamentals_for_universe
from models.features import build_features
from models.ml import train_and_predict_all_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - BACKCOMPAT - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pipeline","ingest","universe","fundamentals","features","train"], default="pipeline")
    p.add_argument("--days", type=int, default=7)
    args = p.parse_args()
    if args.mode == "pipeline":
        sync_broker = os.getenv("SYNC_TO_BROKER", "false").lower() == "true"
        ok = run_eod_pipeline(sync_broker=sync_broker)
        raise SystemExit(0 if ok else 1)
    elif args.mode == "ingest":
        ingest_bars_for_universe(days=args.days)
    elif args.mode == "universe":
        rebuild_universe()
    elif args.mode == "fundamentals":
        fetch_fundamentals_for_universe()
    elif args.mode == "features":
        build_features()
    elif args.mode == "train":
        train_and_predict_all_models()
if __name__ == "__main__":
    main()
