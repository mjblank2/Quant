# data_ingestion/dashboard.py
import sys
import os
from datetime import date, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import sqlalchemy
from sqlalchemy import text
import streamlit as st

# Ensure repo root on sys.path so `db.py` can be imported from a subfolder
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add repo root

st.set_page_config(page_title="Blank Capital Quant – Pro Dashboard", layout="wide")

st.title("Blank Capital Quant")
st.caption("Interactive dashboard for Trades, Positions, Predictions, and Prices (auto-adapts to your schema).")

def _normalize_dsn(url: str) -> str:
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url

@st.cache_resource
def get_engine():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        st.error("DATABASE_URL is not set in your Render service. Set it and refresh.")
        return None
    db_url = _normalize_dsn(db_url)
    try:
        engine = sqlalchemy.create_engine(db_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return engine
    except Exception as e:
        st.error(f"Failed to connect to DB: {e}")
        st.code(db_url, language="text")
        return None

engine = get_engine()
if engine is None:
    st.stop()

# Initialize schema if this is a fresh DB (lazy import to avoid crashing on missing envs)
try:
    from db import create_tables
    try:
        create_tables()
    except Exception as e:
        st.warning(f"Schema init skipped: {e}")
except Exception as e:
    st.warning(f"Could not import create_tables: {e}")

@st.cache_data(ttl=120)
def list_tables() -> List[str]:
    q = text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(q).fetchall()]

@st.cache_data(ttl=120)
def table_columns(tbl: str) -> List[str]:
    q = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = :tbl
        ORDER BY ordinal_position
    """)
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(q, {"tbl": tbl}).fetchall()]

def first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None

def has_table(name: str) -> bool:
    return name in list_tables()

def _download_button_csv(df: pd.DataFrame, label: str, filename: str, key: str = None):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        width="stretch",
        key=key
    )

@st.cache_data(ttl=120)
def load_symbols(limit: int = 2000) -> List[str]:
    universe = []
    with engine.connect() as conn:
        if has_table("daily_bars"):
            q = text("SELECT DISTINCT symbol FROM daily_bars ORDER BY symbol LIMIT :lim")
            universe = [r[0] for r in conn.execute(q, {"lim": limit}).fetchall()]
        elif has_table("trades"):
            q = text("SELECT DISTINCT symbol FROM trades ORDER BY symbol LIMIT :lim")
            universe = [r[0] for r in conn.execute(q, {"lim": limit}).fetchall()]
        elif has_table("predictions"):
            q = text("SELECT DISTINCT symbol FROM predictions ORDER BY symbol LIMIT :lim")
            universe = [r[0] for r in conn.execute(q, {"lim": limit}).fetchall()]
    return universe

with st.sidebar:
    st.header("Global Filters")
    symbols = load_symbols()
    pick_all = st.checkbox("All symbols", value=not symbols)
    selected_symbols = symbols if pick_all else st.multiselect("Symbols", options=symbols, default=symbols[:10] if symbols else [])
    default_start = date.today() - timedelta(days=365)
    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=date.today())

    st.markdown("---")
    st.caption("Filters apply where the view has matching columns (e.g., symbol, ts).")

def _apply_symbol_filter(base_sql: str, cols: List[str]) -> Tuple[str, dict]:
    params = {}
    if selected_symbols and "symbol" in cols:
        base_sql += " AND symbol = ANY(:symbols)"
        params["symbols"] = selected_symbols
    return base_sql, params

def _apply_date_filter(base_sql: str, cols: List[str]) -> Tuple[str, dict]:
    params = {}
    date_candidates = ["ts", "date", "as_of", "executed_at", "filled_at", "created_at", "timestamp", "trade_date"]
    date_col = first_existing(cols, date_candidates)
    if date_col:
        base_sql += f" AND {date_col} BETWEEN :start_d AND :end_d"
        params["start_d"] = start_date
        params["end_d"] = end_date
    return base_sql, params

def _read_df(sql: str, params: dict) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)

def view_prices():
    if not has_table("daily_bars"):
        st.info("Table `daily_bars` not found.")
        return
    cols = table_columns("daily_bars")
    needed = [c for c in ["ts", "symbol", "open", "high", "low", "close", "volume"] if c in cols]
    if not needed:
        st.info("No displayable columns in `daily_bars`.")
        return
    sql = f"SELECT {', '.join(needed)} FROM daily_bars WHERE 1=1"
    sql, p1 = _apply_symbol_filter(sql, cols)
    sql, p2 = _apply_date_filter(sql, cols)
    params = {**p1, **p2}
    sql += " ORDER BY ts, symbol LIMIT 100000"
    df = _read_df(sql, params)
    if df.empty:
        st.info("No price data for current filters.")
        return
    with st.expander("Data (prices)", expanded=False):
        st.dataframe(df, width="stretch", height=350)
        _download_button_csv(df, "Download CSV (prices)", "daily_bars.csv", key="download_prices")

    if "symbol" in df.columns and "close" in df.columns and "ts" in df.columns:
        chart_symbol = st.selectbox("Chart symbol", options=sorted(df["symbol"].unique().tolist()))
        df_sym = df[df["symbol"] == chart_symbol].set_index("ts").sort_index()
        st.line_chart(df_sym["close"], width="stretch")

def view_trades():
    if not has_table("trades"):
        st.info("Table `trades` not found.")
        return
    cols = table_columns("trades")
    sel_cols = [c for c in ["id","client_order_id","symbol","side","quantity","qty","notional","price","status",
                            "executed_at","filled_at","created_at","ts","trade_date","venue"] if c in cols]
    if not sel_cols:
        st.info("No displayable columns in `trades`.")
        return

    sql = f"SELECT {', '.join(sel_cols)} FROM trades WHERE 1=1"
    sql, p1 = _apply_symbol_filter(sql, cols)
    sql, p2 = _apply_date_filter(sql, cols)
    params = {**p1, **p2}

    c1, c2, c3 = st.columns(3)
    with c1:
        side = st.selectbox("Side", options=["(any)","buy","sell"] if "side" in cols else ["(n/a)"])
    with c2:
        status = st.selectbox("Status", options=["(any)","new","filled","partially_filled","canceled","rejected"] if "status" in cols else ["(n/a)"])
    with c3:
        limit_n = st.number_input("Max rows", min_value=100, max_value=200000, value=20000, step=100)

    if "side" in cols and side != "(any)":
        sql += " AND side = :side"
        params["side"] = side
    if "status" in cols and status != "(any)":
        sql += " AND status = :status"
        params["status"] = status

    order_candidates = ["executed_at", "filled_at", "created_at", "ts", "trade_date"]
    order_col = first_existing(cols, order_candidates) or "ts"
    sql += f" ORDER BY {order_col} DESC LIMIT :lim"
    params["lim"] = int(limit_n)

    df = _read_df(sql, params)
    if df.empty:
        st.info("No trades for current filters.")
        return

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Trades", f"{len(df):,}")
    with k2:
        qcol = "quantity" if "quantity" in df.columns else ("qty" if "qty" in df.columns else None)
        if qcol:
            try:
                st.metric("Shares", f"{int(pd.to_numeric(df[qcol], errors='coerce').abs().sum()):,}")
            except Exception:
                st.metric("Shares", "—")
        else:
            st.metric("Shares", "—")
    with k3:
        if "notional" in df.columns:
            try:
                st.metric("Notional", f"${pd.to_numeric(df['notional'], errors='coerce').abs().sum():,.0f}")
            except Exception:
                st.metric("Notional", "—")
        elif qcol and "price" in df.columns:
            try:
                notional = (pd.to_numeric(df[qcol], errors="coerce").abs() * pd.to_numeric(df["price"], errors="coerce").abs()).sum()
                st.metric("Notional", f"${notional:,.0f}")
            except Exception:
                st.metric("Notional", "—")
        else:
            st.metric("Notional", "—")
    with k4:
        if "status" in df.columns:
            st.metric("Filled", f"{(df['status'].astype(str)=='filled').mean()*100:.1f}%")
        else:
            st.metric("Filled", "—")

    st.dataframe(df, width="stretch", height=420)
    _download_button_csv(df, "Download CSV (trades)", "trades.csv", key="download_trades")

def view_positions():
    if not has_table("positions"):
        st.info("Table `positions` not found.")
        return
    cols = table_columns("positions")
    sel_cols = [c for c in ["symbol","shares","quantity","qty","avg_price","market_value","unrealized_pnl",
                            "realized_pnl","side","ts","as_of","updated_at","weight","exposure"] if c in cols]
    if not sel_cols:
        st.info("No displayable columns in `positions`.")
        return

    sql = f"SELECT {', '.join(sel_cols)} FROM positions WHERE 1=1"
    sql, p1 = _apply_symbol_filter(sql, cols)
    sql, p2 = _apply_date_filter(sql, cols)
    params = {**p1, **p2}

    pos_order_candidates = ["as_of", "ts", "updated_at"]
    pos_order_col = first_existing(cols, pos_order_candidates) or "ts"
    sql += f" ORDER BY {pos_order_col} DESC, symbol LIMIT 100000"

    df = _read_df(sql, params)
    if df.empty:
        st.info("No positions for current filters.")
        return

    if "shares" in df.columns:
        qcol = "shares"
    elif "quantity" in df.columns:
        qcol = "quantity"
    elif "qty" in df.columns:
        qcol = "qty"
    else:
        qcol = None

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Positions", f"{df['symbol'].nunique():,}" if "symbol" in df.columns else f"{len(df):,}")
    with k2:
        if qcol:
            try:
                st.metric("Gross Shares", f"{int(pd.to_numeric(df[qcol], errors='coerce').abs().sum()):,}")
            except Exception:
                st.metric("Gross Shares", "—")
        else:
            st.metric("Gross Shares", "—")
    with k3:
        if "market_value" in df.columns:
            try:
                st.metric("Gross MV", f"${pd.to_numeric(df['market_value'], errors='coerce').abs().sum():,.0f}")
            except Exception:
                st.metric("Gross MV", "—")
        else:
            st.metric("Gross MV", "—")
    with k4:
        if qcol:
            try:
                net_shares = pd.to_numeric(df[qcol], errors="coerce").sum()
                st.metric("Net Shares", f"{int(net_shares):,}")
            except Exception:
                st.metric("Net Shares", "—")
        else:
            st.metric("Net Shares", "—")

    st.dataframe(df, width="stretch", height=420)
    _download_button_csv(df, "Download CSV (positions)", "positions.csv", key="download_positions")

def view_predictions():
    if not has_table("predictions"):
        st.info("Table `predictions` not found.")
        return
    cols = table_columns("predictions")
    sel_cols = [c for c in ["ts","symbol","model_version","horizon","score","prediction","rank","prob","y_pred"] if c in cols]
    if not sel_cols:
        st.info("No displayable columns in `predictions`.")
        return

    sql = f"SELECT {', '.join(sel_cols)} FROM predictions WHERE 1=1"
    sql, p1 = _apply_symbol_filter(sql, cols)
    sql, p2 = _apply_date_filter(sql, cols)
    params = {**p1, **p2}

    if "model_version" in cols:
        mv = _read_df("SELECT DISTINCT model_version FROM predictions ORDER BY 1 LIMIT 200", {}).dropna()
        mv_opt = ["(any)"] + mv["model_version"].astype(str).tolist()
        choice = st.selectbox("Model version", options=mv_opt)
        if choice != "(any)":
            sql += " AND model_version = :mv"
            params["mv"] = choice

    pred_order_candidates = ["ts", "symbol"]
    pred_order_col = first_existing(cols, pred_order_candidates) or "ts"
    sql += f" ORDER BY {pred_order_col} DESC, symbol LIMIT 200000"

    df = _read_df(sql, params)
    if df.empty:
        st.info("No predictions for current filters.")
        return

    if "ts" in df.columns:
        latest_date = pd.to_datetime(df["ts"]).max().date()
        pick_date = st.date_input("Cross-section date", value=latest_date)
        cross = df[pd.to_datetime(df["ts"]).dt.date == pick_date].copy()
    else:
        cross = df.copy()

    score_col = None
    for c in ["y_pred","score","prediction","prob","rank"]:
        if c in cross.columns:
            score_col = c
            break

    if score_col:
        st.subheader(f"Top/Bottom by {score_col}")
        cross2 = cross.dropna(subset=[score_col]).copy()
        if cross2.empty:
            st.info("No non-null scores for the selected date.")
        else:
            left, right = st.columns(2)
            with left:
                st.markdown("**Top 20**")
                st.dataframe(cross2.sort_values(score_col, ascending=False).head(20), width="stretch", height=350)
            with right:
                st.markdown("**Bottom 20**")
                st.dataframe(cross2.sort_values(score_col, ascending=True).head(20), width="stretch", height=350)

    with st.expander("All predictions (filtered)", expanded=False):
        st.dataframe(df, width="stretch", height=380)
        _download_button_csv(df, "Download CSV (predictions)", "predictions.csv", key="download_predictions")

tabs = st.tabs(["Overview", "Trades", "Positions", "Predictions", "Prices"])

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Tables in DB", f"{len(list_tables()):,}")
    with c2:
        st.metric("Symbols detected", f"{len(load_symbols()):,}")
    with c3:
        st.metric("Date window", f"{start_date} → {end_date}")
    st.markdown("Use the tabs above and the Global Filters in the sidebar to slice your data.")

with tabs[1]:
    view_trades()

with tabs[2]:
    view_positions()

with tabs[3]:
    view_predictions()

with tabs[4]:
    view_prices()
