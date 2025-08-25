from __future__ import annotations
import streamlit as st, pandas as pd
from sqlalchemy import text
from db import engine
from models.regime import realized_vol, liquidity_breadth, classify_regime

def _df(sql: str, params=None, parse_dates=None):
    with engine.connect() as con:
        try:
            return pd.read_sql_query(text(sql), con, params=params or {}, parse_dates=parse_dates or [])
        except Exception:
            return pd.DataFrame()

def app_panel_v17():
    st.header("🧭 v17: Survivorship + Events + Regime + QP")
    # Regime snapshot
    rv = realized_vol("IWM", 21)
    liq = liquidity_breadth(21)
    regime = classify_regime(rv, liq)
    st.metric("Regime", regime)
    st.caption(f"Realized vol (21d): {rv:.3%} | Median ADV: ${liq:,.0f}" if rv and liq else "Regime metrics unavailable.")

    st.subheader("Option Overlays (latest)")
    ov = _df("SELECT * FROM option_overlays ORDER BY ts DESC, strategy LIMIT 50", parse_dates=['ts','expiry'])
    st.dataframe(ov)

    st.subheader("Recent AltSignals")
    alt = _df("""
        SELECT symbol, ts, name, value FROM alt_signals
        WHERE ts >= (SELECT COALESCE(MAX(ts), '1900-01-01') - INTERVAL '30 days' FROM alt_signals)
        ORDER BY ts DESC
    """, parse_dates=['ts'])
    st.dataframe(alt.head(200))

if __name__ == "__main__":
    app_panel_v17()