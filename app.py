import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import text
from db import create_tables, engine
from data.universe import rebuild_universe
from data.ingest import ingest_bars_for_universe
from data.fundamentals import fetch_fundamentals_for_universe
from models.features import build_features
from models.ml import train_and_predict_all_models, run_walkforward_backtest
from trading.generate_trades import generate_today_trades
from trading.broker import sync_trades_to_broker

st.set_page_config(page_title="Small-Cap Quant System", layout="wide")
st.title("📈 Small-Cap Quant System – Live")

try:
    create_tables()
except Exception as e:
    st.error(f"DB init failed: {e}")
    st.stop()

_ALLOWED_TABLES = {"daily_bars","features","predictions","backtest_equity","universe","trades","positions"}

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
        return pd.read_sql_query(text("SELECT id, trade_date, symbol, side, quantity, price, status, broker_order_id, client_order_id, ts FROM trades ORDER BY id DESC LIMIT 200"), con, parse_dates=["ts","trade_date"])

@st.cache_data(ttl=60)
def load_symbols():
    with engine.connect() as con:
        df = pd.read_sql_query(text("SELECT DISTINCT symbol FROM daily_bars ORDER BY symbol"), con)
    return df["symbol"].tolist()

colA, colB, colC = st.columns(3)
colA.metric("Last Price Date", str(_max_ts("daily_bars")))
colB.metric("Last Features Date", str(_max_ts("features")))
colC.metric("Last Prediction Date", str(_max_ts("predictions")))

with st.sidebar:
    st.header("Controls")
    if st.button("🔁 Rebuild Universe"):
        with st.spinner("Rebuilding universe (Alpaca + Polygon)..."):
            try:
                uni = rebuild_universe()
                st.toast(f"Universe size: {len(uni)}", icon="✅")
            except Exception as e:
                st.error(f"Universe rebuild failed: {e}")

    days = st.number_input("Backfill Days", min_value=30, max_value=3650, value=730, step=30)
    if st.button("⬇️ Backfill Market Data"):
        with st.spinner("Ingesting market data (Alpaca SIP → Polygon → Tiingo)..."):
            try:
                ingest_bars_for_universe(int(days))
                st.toast("Ingestion complete.", icon="✅")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    if st.button("📊 Ingest Fundamentals (Polygon PIT)"):
        with st.spinner("Fetching fundamentals (PIT)..."):
            try:
                df = fetch_fundamentals_for_universe()
                st.toast(f"Fundamentals rows upserted: {len(df)}", icon="✅")
            except Exception as e:
                st.error(f"Fundamentals ingest failed: {e}")

    if st.button("🧱 Build Features (incremental)"):
        with st.spinner("Building features incrementally (batched)..."):
            try:
                feat = build_features()
                st.toast(f"New feature rows: {len(feat):,}", icon="✅")
            except Exception as e:
                st.error(f"Features failed: {e}")

    if st.button("🤖 Train & Predict (all models + blend)"):
        with st.spinner("Training models (XGB / RF / Ridge) & blending..."):
            try:
                outs = train_and_predict_all_models()
                total = sum(len(v) for v in outs.values()) if outs else 0
                st.toast(f"Wrote predictions for {len(outs)} model(s). Total rows: {total:,}", icon="✅")
            except Exception as e:
                st.error(f"Training/predict failed: {e}")

    if st.button("🧪 Run Walk-Forward Backtest (full)"):
        with st.spinner("Running walk-forward backtest with overlapping tranches..."):
            try:
                bt = run_walkforward_backtest()
                st.session_state["backtest"] = bt
                st.toast(f"Backtest rows: {len(bt):,}", icon="✅")
            except Exception as e:
                st.error(f"Backtest failed: {e}")

    if st.button("🧾 Generate Today's Trades & Targets"):
        with st.spinner("Generating trades + writing today's target positions..."):
            try:
                trades = generate_today_trades()
                st.session_state["generated_trades"] = trades
                st.toast(f"Generated trades: {len(trades)}", icon="✅")
            except Exception as e:
                st.error(f"Trade generation failed: {e}")

    if st.button("🔗 Sync with Broker (Alpaca)"):
        with st.spinner("Submitting to broker..."):
            try:
                with engine.connect() as con:
                    recent = pd.read_sql_query(text("SELECT id FROM trades WHERE status='generated' ORDER BY id DESC LIMIT 2000"), con)
                ids = recent["id"].tolist()
                if not ids:
                    st.warning("No 'generated' trades to submit.")
                else:
                    res = sync_trades_to_broker(ids)
                    st.toast(f"Submitted {len(res)} trades.", icon="✅")
            except Exception as e:
                st.error(f"Broker sync failed: {e}")

st.subheader("Universe Summary")
try:
    uni_df = load_universe()
    st.write(f"Universe size: {len(uni_df)}")
    st.dataframe(uni_df.head(50))
except Exception:
    st.info("Universe table empty or not available yet.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Latest Prices (Pick a Symbol)")
    try:
        syms = load_symbols()
        if syms:
            sym = st.selectbox("Symbol", syms, index=0)
            with engine.connect() as con:
                prices_df = pd.read_sql_query(
                    text("SELECT ts, COALESCE(adj_close, close) AS close FROM daily_bars WHERE symbol = :symbol ORDER BY ts DESC LIMIT 504"),
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
    st.dataframe(trades)
    st.download_button("Download Trades CSV", trades.to_csv(index=False).encode(), "trades.csv", "text/csv")
except Exception:
    st.info("No trades yet.")
