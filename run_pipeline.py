from __future__ import annotations
import os
import logging
from data.ingest import ingest_bars_for_universe
from data.fundamentals import fetch_fundamentals_for_universe
from models.features import build_features
from models.ml import train_and_predict_all_models
from trading.generate_trades import generate_today_trades
from trading.broker import sync_trades_to_broker


def main(sync_broker: bool = False):
    ingest_bars_for_universe(7)
    fetch_fundamentals_for_universe()
    build_features()
    outs = train_and_predict_all_models()
    trades = generate_today_trades()
    if sync_broker:
        ids = trades.index.tolist() if "id" in trades.columns else []
        if ids:
            sync_trades_to_broker(ids)
    return True


if __name__ == "__main__":
    do_sync = os.getenv("SYNC_TO_BROKER", "false").lower() == "true"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Starting pipeline (sync_to_broker=%s)", do_sync)
    main(sync_broker=do_sync)
