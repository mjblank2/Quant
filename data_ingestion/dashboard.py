# data_ingestion/dashboard.py
import os
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import sqlalchemy
from sqlalchemy import text

# --- Database Connection ---
@st.cache_resource
def get_engine():
    try:
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            st.error("DATABASE_URL environment variable is not set.")
            return None
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)
        return sqlalchemy.create_engine(db_url, pool_pre_ping=True)
    except Exception as e:
        st.error(f"Failed to create database engine: {e}")
        return None

# --- Data Fetching ---
@st.cache_data(ttl=600)
def get_symbols():
    engine = get_engine()
    if engine is None:
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT DISTINCT symbol FROM daily_bars ORDER BY symbol")).fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        st.warning(f"Could not load symbols: {e}")
        return []

@st.cache_data(ttl=600)
def get_daily_bars(symbol: str):
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT ts, close
                FROM daily_bars
                WHERE symbol = :symbol AND ts > NOW() - INTERVAL '2 years'
                ORDER BY ts;
            """)
            df = pd.read_sql_query(query, conn, params={"symbol": symbol}, index_col="ts")
            return df
    except Exception as e:
        st.warning(f"Could not load data for {symbol}: {e}")
        return pd.DataFrame()

def get_current_positions():
    # TODO: replace with real positions table when available
    data = {'symbol': ['AAPL', 'MSFT', 'PLTR'],
            'shares': [100, 200, 500],
            'market_value': [17500, 62000, 12500],
            'daily_pnl': [500, -200, 150]}
    return pd.DataFrame(data)

def get_trade_log():
    # TODO: replace with real transactions table when available
    data = {'timestamp': pd.to_datetime(['2025-08-20 10:05', '2025-08-21 14:20']),
            'symbol': ['AAPL', 'MSFT'],
            'action': ['BUY', 'SELL'],
            'quantity': [100, 50],
            'price': [175.00, 310.50]}
    return pd.DataFrame(data)

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# --- Streamlit App ---
st.set_page_config(page_title="Quantitative Trading Dashboard", layout="wide")
st.title("📈 Quantitative Trading System Dashboard")

# Sidebar controls (stubs for now)
st.sidebar.header("System Controls")
if st.sidebar.button("▶️ Run Full Backtest"):
    st.sidebar.info("Triggering backtest job…")
if st.sidebar.button("📊 Generate Today's Trades"):
    st.sidebar.info("Triggering signal generation DAG…")
if st.sidebar.button("🔄 Sync with Broker"):
    st.sidebar.info("Sending reconciliation request…")

# Data export
st.sidebar.header("Data Export")
trade_log_df = get_trade_log()
st.sidebar.download_button(
    label="📥 Download Trade Log (Excel)",
    data=to_excel(trade_log_df),
    file_name="trade_log.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# Main
st.header("Portfolio Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total P&L", "$50,123.45", "1.2%")
col2.metric("Annualized Return", "35.2%", "5.2%")
col3.metric("Sharpe Ratio", "2.15")
col4.metric("Max Drawdown", "-12.5%")

st.subheader("Equity Curve")
symbols = get_symbols()
selected_symbol = st.selectbox("Select Symbol to View Chart", options=symbols or ["AAPL"])
if selected_symbol:
    bars_df = get_daily_bars(selected_symbol)
    if not bars_df.empty:
        fig_equity = px.line(bars_df, y='close', title=f'{selected_symbol} Price History')
        st.plotly_chart(fig_equity, use_container_width=True)
    else:
        st.warning(f"No data found for {selected_symbol}.")

st.subheader("Current Positions")
st.dataframe(get_current_positions(), use_container_width=True)
