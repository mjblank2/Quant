from __future__ import annotations
import streamlit as st
import os
import pandas as pd
import plotly.express as px
from sqlalchemy import text, bindparam
from db import create_tables, engine
from data.universe import rebuild_universe
from data.ingest import ingest_bars_for_universe
from data.fundamentals import fetch_fundamentals_for_universe
from models.features import build_features
from utils.price_utils import select_price_as
from models.ml import train_and_predict_all_models, run_walkforward_backtest
from trading.generate_trades import generate_today_trades
from trading.broker import sync_trades_to_broker

# Task queue imports
try:
    from tasks import (
        rebuild_universe_task, ingest_market_data_task, ingest_fundamentals_task,
        build_features_task, train_and_predict_task, run_backtest_task,
        generate_trades_task, sync_broker_task, run_full_pipeline_task
    )
    from tasks.task_utils import dispatch_task, get_task_status, get_recent_tasks
    TASK_QUEUE_AVAILABLE = True
except ImportError:
    TASK_QUEUE_AVAILABLE = False

st.set_page_config(page_title="Small-Cap Quant System", layout="wide")
st.title("📈 Small-Cap Quant System – Live (Polygon PTI + Ensemble)")

try:
    create_tables()
except Exception as e:
    st.error(f"DB init failed: {e}")
    st.stop()

_ALLOWED_TABLES = {"daily_bars","features","predictions","backtest_equity","universe","trades","positions","fundamentals"}

def _max_ts(table: str):
    if table not in _ALLOWED_TABLES:
        return None
    try:
        with engine.connect() as con:
            return con.execute(text(f"SELECT MAX(ts) FROM {table}")).scalar()
    except Exception:
        return None

@st.cache_data(ttl=900)
def load_universe():
    with engine.connect() as con:
        return pd.read_sql_query(text("SELECT * FROM universe ORDER BY symbol LIMIT 10000"), con)

@st.cache_data(ttl=60)
def load_trades():
    with engine.connect() as con:
        return pd.read_sql_query(text(
            "SELECT id, trade_date, symbol, side, quantity, price, status, broker_order_id, client_order_id, ts "
            "FROM trades ORDER BY id DESC LIMIT 200"
        ), con, parse_dates=["ts", "trade_date"])

@st.cache_data(ttl=60)
def load_symbols():
    with engine.connect() as con:
        df = pd.read_sql_query(text("SELECT DISTINCT symbol FROM daily_bars ORDER BY symbol"), con)
    return df["symbol"].tolist()

colA, colB, colC = st.columns(3)
colA.metric("Last Price Date", str(_max_ts("daily_bars")))
colB.metric("Last Features Date", str(_max_ts("features")))
colC.metric("Last Prediction Date", str(_max_ts("predictions")))

# Main-page quick actions so users can run the pipeline even if the
# sidebar is hidden in their Streamlit deployment.
st.subheader("Quick Actions")
use_task_queue_main = st.checkbox(
    "Use Task Queue (async)",
    value=TASK_QUEUE_AVAILABLE,
    key="main_use_task_queue",
    disabled=not TASK_QUEUE_AVAILABLE,
)
if st.button("🚀 Run Full Pipeline", key="main_run_pipeline"):
    if use_task_queue_main and TASK_QUEUE_AVAILABLE:
        try:
            task_id = dispatch_task(run_full_pipeline_task, False)
            st.success(f"Full pipeline task dispatched: {task_id[:8]}...")
            st.session_state[f"task_{task_id}"] = {"name": "Full Pipeline", "id": task_id}
        except Exception as e:
            st.error(f"Failed to dispatch task: {e}")
    else:
        try:
            from run_pipeline import main as run_full_pipeline_main  # type: ignore
        except Exception:
            run_full_pipeline_main = None
        if run_full_pipeline_main is None:
            st.error("Pipeline module is unavailable. Ensure run_pipeline.py exists in your project.")
        else:
            with st.spinner("Running full pipeline (this may take several minutes)…"):
                try:
                    success = run_full_pipeline_main(sync_broker=False)
                    if success:
                        st.success("Pipeline completed successfully.")
                    else:
                        st.error("Pipeline completed with errors. Check logs for details.")
                except Exception as e:
                    st.error(f"Pipeline execution failed: {e}")

st.divider()

with st.sidebar:
    st.header("Controls")
    # Toggle between task queue and direct execution
    use_task_queue = st.checkbox(
        "Use Task Queue (async)", value=TASK_QUEUE_AVAILABLE, disabled=not TASK_QUEUE_AVAILABLE
    )
    if not TASK_QUEUE_AVAILABLE and use_task_queue:
        st.warning("Task queue not available. Install Redis and run Celery worker.")

    if st.button("🔁 Rebuild Universe"):
        if use_task_queue and TASK_QUEUE_AVAILABLE:
            try:
                task_id = dispatch_task(rebuild_universe_task)
                st.success(f"Universe rebuild task dispatched: {task_id[:8]}...")
                st.session_state[f"task_{task_id}"] = {"name": "Rebuild Universe", "id": task_id}
            except Exception as e:
                st.error(f"Failed to dispatch task: {e}")
        else
            with st.spinner("Rebuilding universe (Polygon)..."):
                try:
                    uni = rebuild_universe()
                    st.toast(f"Universe size: {len(uni)}", icon="✅")
                except Exception as e:
                    st.error(f"Universe rebuild failed: {e}")

    days = st.number_input("Backfill Days", min_value=30, max_value=3650, value=730, step=30)
    if st.button("⬇️ Backfill Market Data"):
        if use_task_queue and TASK_QUEUE_AVAILABLE:
            try:
                task_id = dispatch_task(ingest_market_data_task, int(days))
                st.success(f"Market data ingestion task dispatched: {task_id[:8]}...")
                st.session_state[f"task_{task_id}"] = {"name": "Backfill Market Data", "id": task_id}
            except Exception as e:
                st.error(f"Failed to dispatch task: {e}")
        else:
            with st.spinner("Ingesting market data (Polygon -> Tiingo)..."):
                try:
                    ingest_bars_for_universe(int(days))
                    st.toast("Ingestion complete.", icon="✅")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    if st.button("📊 Ingest Fundamentals (Polygon PTI)"):
        if use_task_queue and TASK_QUEUE_AVAILABLE:
            try:
                task_id = dispatch_task(ingest_fundamentals_task)
                st.success(f"Fundamentals ingestion task dispatched: {task_id[:8]}...")
                st.session_state[f"task_{task_id}"] = {"name": "Ingest Fundamentals", "id": task_id}
            except Exception as e:
                st.error(f"Failed to dispatch task: {e}")
        else:
            with st.spinner("Fetching fundamentals (point-in-time from Polygon)..."):
                try:
                    df = fetch_fundamentals_for_universe()
                    st.toast(f"Fundamentals rows upserted: {len(df)}", icon="✅")
                except Exception as e:
                    st.error(f"Fundamentals ingest failed: {e}")

    if st.button("🧱 Build Features (incremental)"):
        if use_task_queue and TASK_QUEUE_AVAILABLE:
            try:
                task_id = dispatch_task(build_features_task)
                st.success(f"Feature building task dispatched: {task_id[:8]}...")
                st.session_state[f"task_{task_id}"] = {"name": "Build Features", "id": task_id}
            except Exception as e:
                st.error(f"Failed to dispatch task: {e}")
        else:
            with st.spinner("Building features incrementally (batched)…"):
                try:
                    feat = build_features()
                    st.toast(f"New feature rows: {len(feat):,}", icon="✅")
                except Exception as e:
                    st.error(f"Features failed: {e}")

    if st.button("🤖 Train & Predict (XGB / RF / Ridge + blend)"):
        if use_task_queue and TASK_QUEUE_AVAILABLE:
            try:
                task_id = dispatch_task(train_and_predict_task)
                st.success(f"Training task dispatched: {task_id[:8]}…")
                st.session_state[f"task_{task_id}"] = {"name": "Train & Predict", "id": task_id}
            except Exception as e:
                st.error(f"Failed to dispatch task: {e}")
        else:
            with st.spinner("Training models & blending…"):
                try:
                    outs = train_and_predict_all_models()
                    total = sum(len(v) for v in outs.values()) if outs else 0
                    st.toast(f"Wrote predictions for {len(outs)} model(s). Total rows: {total:,}", icon="✅")
                except Exception as e:
                    st.error(f"Training/predict failed: {e}")

    if st.button("🧪 Walk-Forward Backtest"):
        if use_task_queue and TASK_QUEUE_AVAILABLE:
            try:
                task_id = dispatch_task(run_backtest_task)
                st.success(f"Backtest task dispatched: {task_id[:8]}…")
                st.session_state[f"task_{task_id}"] = {"name": "Walk-Forward Backtest", "id": task_id}
            except Exception as e:
                st.error(f"Failed to dispatch task: {e}")
        else:
            with st.spinner("Running walk-forward backtest with exposure scaling…"):
                try:
                    bt = run_walkforward_backtest()
                    st.session_state["backtest"] = bt
                    st.toast(f"Backtest rows: {len(bt):,}", icon="✅")
                except Exception as e:
                    st.error(f"Backtest failed: {e}")

    if st.button("🧾 Generate Today's Trades & Targets"):
        if use_task_queue and TASK_QUEUE_AVAILABLE:
            try:
                task_id = dispatch_task(generate_trades_task)
                st.success(f"Trade generation task dispatched: {task_id[:8]}…")
                st.session_state[f"task_{task_id}"] = {"name": "Generate Trades", "id": task_id}
            except Exception as e:
                st.error(f"Failed to dispatch task: {e}")
        else:
            with st.spinner("Generating trades + writing today's target positions…"):
                try:
                    trades = generate_today_trades()
                    st.session_state["generated_trades"] = trades
                    st.toast(f"Generated trades: {len(trades)}", icon="✅")
                except Exception as e:
                    st.error(f"Trade generation failed: {e}")

    if st.button("🔗 Sync with Broker (Interactive Brokers)"):
        if use_task_queue and TASK_QUEUE_AVAILABLE:
            try:
                task_id = dispatch_task(sync_broker_task)
                st.success(f"Broker sync task dispatched: {task_id[:8]}…")
                st.session_state[f"task_{task_id}"] = {"name": "Sync with Broker", "id": task_id}
            except Exception as e:
                st.error(f"Failed to dispatch task: {e}")
        else:
            with st.spinner("Submitting to broker…"):
                try:
                    with engine.connect() as con:
                        recent = pd.read_sql_query(
                            text("SELECT id FROM trades WHERE status='generated' ORDER BY id DESC LIMIT 2000"),
                            con
                        )
                    ids = recent["id"].tolist()
                    if not ids:
                    res = sync_trades_to_broker(ids, host=os.getenv("IB_HOST", "127.0.0.1"), port=int(os.getenv("IB_PORT", 7497)), client_id=int(os.getenv("IB_CLIENT_ID", 1)))
