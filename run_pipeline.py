from __future__ import annotations
from db import create_tables
from data.ingest import ingest_bars_for_universe
from data.fundamentals import fetch_fundamentals_for_universe
from models.features import build_features
from models.ml import train_and_predict_all_models
from trading.generate_trades import generate_today_trades
from trading.broker import sync_trades_to_broker
from config import PIPELINE_SYNC_BROKER

def run():
    create_tables()
    ingest_bars_for_universe(7)
    fetch_fundamentals_for_universe()
    build_features()
    train_and_predict_all_models()
    trades = generate_today_trades()
    if PIPELINE_SYNC_BROKER and not trades.empty:
        # submit all generated trades
        from db import engine
        import pandas as pd
        from sqlalchemy import text
        with engine.connect() as con:
            df = pd.read_sql_query(text("SELECT id FROM trades WHERE status='generated' ORDER BY id DESC LIMIT 2000"), con)
        ids = df["id"].tolist()
        if ids:
            sync_trades_to_broker(ids)

if __name__ == "__main__":
    run()
