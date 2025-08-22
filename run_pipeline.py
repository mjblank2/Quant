from __future__ import annotations
import argparse
from data.ingest import ingest_bars_for_universe
from data.fundamentals import fetch_fundamentals_for_universe
from models.features import build_features
from models.ml import train_and_predict_all_models
from trading.generate_trades import generate_today_trades
from trading.broker import sync_trades_to_broker
from db import get_engine
from sqlalchemy import text
from config import PIPELINE_SYNC_BROKER, PIPELINE_BACKFILL_DAYS

def main(days: int, sync_broker: bool):
    ingest_bars_for_universe(days)
    fetch_fundamentals_for_universe()
    build_features()
    train_and_predict_all_models()
    trades = generate_today_trades()
    print(f"Generated trades: {len(trades)}")
    if sync_broker or PIPELINE_SYNC_BROKER:
        with get_engine().connect() as con:
            recent = con.execute(text("SELECT id FROM trades WHERE status='generated' ORDER BY id DESC LIMIT 2000")).mappings().all()
        ids = [r["id"] for r in recent]
        if ids:
            res = sync_trades_to_broker(ids)
            print(f"Submitted {len(res)} trades to broker.")
        else:
            print("No generated trades to submit.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=PIPELINE_BACKFILL_DAYS, help="Backfill window for price ingestion")
    p.add_argument("--sync-broker", action="store_true", help="Submit generated trades to broker")
    args = p.parse_args()
    main(args.days, args.sync_broker)

