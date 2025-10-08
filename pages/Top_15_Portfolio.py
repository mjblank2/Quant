# pages/Top_15_Portfolio.py
from __future__ import annotations

import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import text, bindparam
from db import engine
from trading.top15_portfolio_tracker import get_top_predictions, compute_display_weights

st.set_page_config(page_title="Top‑15 Portfolio (5‑Day)", layout="wide")
st.title("Top‑15 Portfolio (5‑Day)")
st.caption("Liquidity‑screened, sub‑$3B, 5‑day targeted strategy with actionable alerts.")

# --- Controls ---
n = st.slider("Top‑N", 10, 30, 15, 5)
min_adv = st.number_input("Min ADV (USD, 21d)", min_value=100_000, value=1_000_000, step=100_000)
max_mcap = st.number_input("Max Market Cap (USD)", min_value=100_000_000, value=3_000_000_000, step=100_000_000)

# --- Load top‑N today ---
try:
    top_df = get_top_predictions(n=n, min_adv=float(min_adv), max_market_cap=float(max_mcap))
except Exception as e:
    st.error(f"Could not load predictions/features: {e}")
    top_df = pd.DataFrame(columns=["symbol","y_pred","adv_usd_21","market_cap","size_ln"])

if not top_df.empty:
    top_df["weight_display"] = compute_display_weights(top_df).round(4)

# --- Show portfolio table + chart ---
st.subheader("Current Top‑N (after filters)")
if top_df.empty:
    st.info("No eligible names today (check data & filters).")
else:
    st.dataframe(
        top_df.set_index("symbol").assign(
            y_pred=lambda d: d["y_pred"].round(4),
            adv_usd_21=lambda d: d["adv_usd_21"].round(0),
            market_cap=lambda d: d["market_cap"].round(0),
        ),
        use_container_width=True,
    )
    fig = px.bar(top_df, x="symbol", y="y_pred", title="Predicted 5‑Day Returns")
    st.plotly_chart(fig, use_container_width=True)

# --- Determine previous holdings (DB if available, else JSON state) ---
def _load_prev_holdings_from_db() -> pd.DataFrame:
    try:
        with engine.connect() as con:
            df = pd.read_sql_query(
                text("SELECT symbol, entry_date FROM top15_holdings ORDER BY symbol"),
                con,
                parse_dates=["entry_date"],
            )
        return df
    except Exception:
        return pd.DataFrame(columns=["symbol","entry_date"])

def _load_prev_holdings_from_json(path="top15_portfolio_state.json") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["symbol","entry_date"])
    try:
        with open(path, "r", encoding="utf-8") as f:
            syms = json.load(f).get("symbols", [])
        return pd.DataFrame({"symbol": syms, "entry_date": pd.NaT})
    except Exception:
        return pd.DataFrame(columns=["symbol","entry_date"])

prev_df = _load_prev_holdings_from_db()
if prev_df.empty:
    prev_df = _load_prev_holdings_from_json()

held = set(prev_df["symbol"].tolist())
now = set(top_df["symbol"].tolist() if not top_df.empty else [])

# --- TTL (5 trading days) helper: find 5‑day‑ago trading day from daily_bars ---
def trading_cutoff_ago(n_days: int = 5):
    try:
        with engine.connect() as con:
            q = text("""
                SELECT ts FROM (
                  SELECT DISTINCT DATE(ts) AS ts
                  FROM daily_bars
                  ORDER BY ts DESC
                  LIMIT :n
                ) q ORDER BY ts LIMIT 1
            """)
            ref = pd.read_sql_query(q, con, params={"n": n_days + 1}, parse_dates=["ts"])
        return None if ref.empty else ref.iloc[0]["ts"].date()
    except Exception:
        return None

cutoff_trading_date = trading_cutoff_ago(5)

# --- Entries/Exits ---
entries = sorted(now - held)
exits_rank = sorted(held - now)

# TTL exits from DB (if we have entry dates + cutoff)
exits_ttl = []
if cutoff_trading_date is not None and not prev_df.empty and "entry_date" in prev_df.columns:
    exits_ttl = prev_df.loc[
        prev_df["entry_date"].notna() & (prev_df["entry_date"].dt.date <= cutoff_trading_date),
        "symbol"
    ].tolist()

exits = sorted(set(exits_rank).union(exits_ttl))

# --- Build alerts tables ---
def _alerts_df(side: str, syms: list[str]) -> pd.DataFrame:
    if not syms:
        return pd.DataFrame(columns=["symbol","reason","suggested_weight"])
    d = top_df[top_df["symbol"].isin(syms)].copy() if not top_df.empty else pd.DataFrame({"symbol": syms})
    d["suggested_weight"] = d.get("weight_display", pd.Series(0.0, index=d.index)).fillna(0.0)
    d["reason"] = "Entered Top‑N" if side == "BUY" else "Dropped from Top‑N/TTL"
    cols = ["symbol","reason","suggested_weight"]
    return d[cols].sort_values("suggested_weight", ascending=False)

buy_df  = _alerts_df("BUY", entries)
sell_df = _alerts_df("SELL", exits)

st.subheader("Actionable Alerts (BUY / SELL)")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**BUY Alerts**")
    st.dataframe(buy_df.set_index("symbol"), use_container_width=True)
with c2:
    st.markdown("**SELL Alerts**")
    st.dataframe(sell_df.set_index("symbol"), use_container_width=True)

# --- Trade intents log (CSV) ---
st.subheader("Top‑15 Trades Log (CSV)")
log_path = "top15_trades_log.csv"
if os.path.exists(log_path):
    log_df = pd.read_csv(log_path)
    st.dataframe(log_df.tail(200), use_container_width=True)
    st.download_button(
        "Download Top‑15 Trades Log",
        log_df.to_csv(index=False).encode(),
        "top15_trades_log.csv",
        "text/csv",
        key="dl_top15_log",
    )
else:
    st.info("No trades log yet. It will appear after the first daily update runs.")
