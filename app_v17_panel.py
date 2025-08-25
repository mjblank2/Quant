from __future__ import annotations
import streamlit as st, pandas as pd
from sqlalchemy import text
from db import engine
from models.regime import classify_regime

def _df(sql: str, params=None, parse_dates=None):
    with engine.connect() as con:
        try:
            return pd.read_sql_query(text(sql), con, params=params or {}, parse_dates=parse_dates or [])
        except Exception:
            return pd.DataFrame()

def app_panel_v17():
    st.header("🧭 v17: Survivorship-safe + PEAD/Russell + Regime-gated + Overlays")
    ts = _df("SELECT MAX(ts) AS ts FROM target_positions")
    ts = ts['ts'].iloc[0] if not ts.empty else None
    if ts:
        st.caption(f"Latest target date: {ts}")
        st.write("Regime:", classify_regime(ts))
    st.subheader("Recent AltSignals")
    alt = _df("SELECT ts, symbol, name, value FROM alt_signals ORDER BY ts DESC LIMIT 200", parse_dates=['ts'])
    st.dataframe(alt)

    st.subheader("Option Overlays (latest)")
    ov = _df("SELECT as_of, symbol, strategy, tenor_days, put_strike, call_strike, est_premium_pct, notes FROM option_overlays ORDER BY as_of DESC LIMIT 200")
    st.dataframe(ov)

if __name__ == "__main__":
    app_panel_v17()
