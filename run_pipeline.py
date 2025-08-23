from __future__ import annotations
import argparse
from data.ingest import ingest_bars_for_universe
from data.fundamentals import fetch_fundamentals_for_universe
from models.features import build_features
from models.ml import train_and_predict_all_models
from trading.generate_trades import generate_today_trades
from trading.broker import sync_trades_to_broker
from config import PIPELINE_SYNC_BROKER
from sqlalchemy import text
from db import engine

def main(days: int = 7):
    ingest_bars_for_universe(days)
    fetch_fundamentals_for_universe()
    build_features()
    outs = train_and_predict_all_models()
    # Generate trades after predictions
    trades_df = generate_today_trades()
    if PIPELINE_SYNC_BROKER:
        with engine.connect() as con:
            res = con.execute(text("SELECT id FROM trades WHERE status='generated' ORDER BY id DESC LIMIT 2000"))
            trade_ids = [r[0] for r in res]
        if trade_ids:
            sync_trades_to_broker(trade_ids)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=7)
    args = p.parse_args()
    main(args.days)
