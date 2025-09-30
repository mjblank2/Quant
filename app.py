from __future__ import annotations
import streamlit as st
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
        else:
            with st.spinner("Rebuilding universe (Alpaca + Polygon)..."):
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
            with st.spinner("Ingesting market data (Alpaca → Polygon → Tiingo)..."):
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

    if st.button("🔗 Sync with Broker (Alpaca)"):
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
                        st.warning("No 'generated' trades to submit.")
                    else:
                        res = sync_trades_to_broker(ids)
                        st.toast(f"Submitted {len(res)} trades.", icon="✅")
                except Exception as e:
                    st.error(f"Broker sync failed: {e}")

    # Full pipeline button
    st.divider()
    if st.button("🚀 Run Full Pipeline"):
        if use_task_queue and TASK_QUEUE_AVAILABLE:
            try:
                task_id = dispatch_task(run_full_pipeline_task, False)  # Don't sync broker by default
                st.success(f"Full pipeline task dispatched: {task_id[:8]}…")
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

st.subheader("Universe Summary")
try:
    uni_df = load_universe()
    st.write(f"Universe size: {len(uni_df)}")
    st.dataframe(uni_df.head(50), width='stretch')
except Exception:
    st.info("Universe table empty or not available yet.")

# Task monitoring section
if TASK_QUEUE_AVAILABLE:
    st.subheader("Task Monitoring")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        try:
            recent_tasks = get_recent_tasks(10)
            if recent_tasks:
                task_df = pd.DataFrame(recent_tasks)
                task_df['duration'] = task_df.apply(lambda row:
                    str(row['completed_at'] - row['started_at']) if row['completed_at'] and row['started_at'] else 'Running',
                    axis=1
                )
                for _, task in task_df.iterrows():
                    status = task['status']
                    if status == 'SUCCESS':
                        st.success(f"✅ {task['task_name']} ({task['task_id'][:8]}) - {task['duration']}")
                    elif status == 'FAILURE':
                        st.error(f"❌ {task['task_name']} ({task['task_id'][:8]}) - {task['error_message']}")
                    elif status == 'STARTED':
                        st.info(f"🔄 {task['task_name']} ({task['task_id'][:8]}) - {task['progress']}%")
                    else:
                        st.warning(f"⏳ {task['task_name']} ({task['task_id'][:8]}) - {status}")
            else:
                st.info("No recent tasks")
        except Exception as e:
            st.error(f"Could not load task status: {e}")

    with col_right:
        if st.button("🔄 Refresh Tasks"):
            st.rerun()
        for key, task_info in st.session_state.items():
            if key.startswith("task_") and isinstance(task_info, dict):
                task_id = task_info.get("id")
                if task_id:
                    status = get_task_status(task_id)
                    if status:
                        with st.expander(f"{task_info['name']} ({task_id[:8]})"):
                            st.write(f"Status: {status['status']}")
                            st.write(f"Progress: {status['progress']}%")
                            if status['result']:
                                st.json(status['result'])
                            if status['error_message']:
                                st.error(status['error_message'])
else:
    st.info("💡 Install Redis and run Celery worker to enable async task queue")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Latest Prices (Pick a Symbol)")
    try:
        syms = load_symbols()
        if syms:
            sym = st.selectbox("Symbol", syms, index=0)
            with engine.connect() as con:
                prices_df = pd.read_sql_query(
                    text(f"SELECT ts, {select_price_as('close')} FROM daily_bars WHERE symbol = :symbol ORDER BY ts DESC LIMIT 504"),
                    con,
                    params={"symbol": sym},
                    parse_dates=["ts"]
                ).sort_values("ts")
            if not prices_df.empty:
                fig = px.line(prices_df, x="ts", y="close", title=f"{sym} Close (last ~2y, adjusted if available)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prices yet. Backfill market data first.")
    except Exception:
        st.info("Prices table empty or not available yet.")

with col2:
    st.subheader("Backtest Equity (if run)")
    bt = st.session_state.get("backtest")
    if bt is None:
        try:
            with engine.connect() as con:
                bt = pd.read_sql_query(text("SELECT * FROM backtest_equity ORDER BY ts"), con, parse_dates=["ts"])
        except Exception:
            bt = None
    if bt is not None and not bt.empty:
        fig2 = px.line(bt, x="ts", y="equity", title="Equity Curve")
        st.plotly_chart(fig2, use_container_width=True)

st.subheader("Trade Log (latest 200)")
try:
    trades = load_trades()
    st.dataframe(trades, width='stretch')
    st.download_button(
        "Download Trades CSV",
        trades.to_csv(index=False).encode(),
        "trades.csv",
        "text/csv",
        key="download_trades_main"
    )
except Exception:
    st.info("No trades yet.")

# =============================================================================
# Enhanced Model Predictions & Suggested Trades Section
# =============================================================================

@st.cache_data(ttl=300)
def load_latest_predictions_with_features(n_top: int = 20) -> pd.DataFrame:
    """
    Load latest predictions for the preferred model and merge with select features.
    Returns the top n rows sorted by predicted return.
    """
    from config import PREFERRED_MODEL
    sql = text("""
        WITH target AS (
            SELECT MAX(ts) AS mx
            FROM predictions
            WHERE model_version = :mv
        )
        SELECT symbol, ts, y_pred
        FROM predictions
        WHERE model_version = :mv
          AND ts = (SELECT mx FROM target)
        ORDER BY y_pred DESC
        LIMIT :limit;
    """)
    with engine.connect() as con:
        preds = pd.read_sql_query(
            sql,
            con,
            params={"mv": PREFERRED_MODEL, "limit": n_top * 5},
        )
    if preds.empty:
        return preds
    syms = preds["symbol"].tolist()
    stmt_feats = text("""
        SELECT symbol, ts, vol_21, adv_usd_21, size_ln, mom_21,
               turnover_21, beta_63
        FROM features
        WHERE symbol IN :syms
          AND ts = (SELECT MAX(ts) FROM features);
    """).bindparams(bindparam("syms", expanding=True))
    with engine.connect() as con:
        feats = pd.read_sql_query(stmt_feats, con, params={"syms": tuple(syms)})
    merged = preds.merge(feats, on="symbol", how="left")
    return merged.sort_values("y_pred", ascending=False).head(n_top)

def compute_display_weights(df: pd.DataFrame) -> pd.Series:
    """Normalise predicted returns to positive weights for display."""
    preds = df["y_pred"].astype(float)
    min_pred = preds.min()
    shifted = preds - min_pred if min_pred < 0 else preds
    total = shifted.sum()
    return shifted / total if total else pd.Series(0.0, index=df.index)

st.subheader("Top Predictions & Suggested Trades")
top_n = st.slider(
    "Select number of top signals to display",
    min_value=5,
    max_value=30,
    value=10,
    step=5,
    help="Controls how many high-confidence names are shown."
)

try:
    preds_df = load_latest_predictions_with_features(top_n)
    if preds_df.empty:
        st.info("No predictions available. Run the training pipeline to generate signals.")
    else:
        preds_df["weight"] = compute_display_weights(preds_df)
        # round numeric columns for display
        for col in preds_df.select_dtypes(include=["float", "int"]).columns:
            preds_df[col] = preds_df[col].round(4)

        st.dataframe(preds_df.set_index("symbol"), use_container_width=True)

        fig_preds = px.bar(
            preds_df,
            x="symbol",
            y="y_pred",
            title="Predicted Returns for Top Signals",
            labels={"y_pred": "Predicted Return", "symbol": "Symbol"},
        )
        st.plotly_chart(fig_preds, use_container_width=True)

        with st.expander("What do these numbers mean?"):
            st.markdown("""
- **Predicted Return (`y_pred`)** – The ensemble model’s forecast of next-period return.  Higher values imply stronger expected outperformance.
- **Weight** – A normalised representation of each signal’s relative strength.  The actual trading algorithm may apply a more sophisticated optimiser, but this gives a sense of conviction.
- **Volatility (21d)** – Rolling 21-day standard deviation of returns.
- **ADV USD (21d)** – Average dollar volume over 21 days (liquidity proxy).
- **Size (ln MV)** – Natural log of market capitalisation.
- **Momentum (21d)** – Short-term momentum indicator.
- **Turnover (21d)** – Volume relative to shares outstanding.
- **Beta (63d)** – Beta versus the market.
""")

        with st.expander("Model & Process Overview"):
            st.markdown("""
The small‑cap quant system uses an ensemble of machine‑learning models—gradient boosting (XGBoost, LightGBM, CatBoost), random forests, neural networks and a simple reinforcement‑learning‑inspired model—to forecast expected returns.  Models are trained on historical price, volume, momentum, volatility and fundamental factors, and blended according to their information coefficient in the current regime.  A risk‑aware optimiser then translates signals into portfolio weights.  This section helps you see not just the final ranking but also the underlying features, so you understand why each stock appears in the list.
""")
except Exception as e:
    st.error(f"Failed to load predictions: {e}")
