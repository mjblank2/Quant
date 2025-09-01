from __future__ import annotations
import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import engine
from risk.risk_model import portfolio_beta
from risk.sector import sector_asof

def _df(sql: str, params=None, parse_dates=None):
    with engine.connect() as con:
        try:
            return pd.read_sql_query(text(sql), con, params=params or {}, parse_dates=parse_dates or [])
        except Exception:
            return pd.DataFrame()

def app_panel_v16():
    st.header("🧭 v16: De-Crowded + Tax-Aware Panel")
    latest_ts = _df("SELECT MAX(ts) AS ts FROM target_positions")
    if latest_ts.empty or latest_ts['ts'].iloc[0] is None:
        st.info("No targets yet.")
        return
    ts = latest_ts['ts'].iloc[0]
    tp = _df("SELECT symbol, weight FROM target_positions WHERE ts=:t ORDER BY weight DESC", {'t': ts})
    if tp.empty:
        st.info("No targets for latest date.")
        return

    st.subheader("Current Target Book")
    st.dataframe(tp, width='stretch')

    # Market-beta estimate
    try:
        w = tp.set_index('symbol')['weight']
        b = portfolio_beta(w, ts)
    except Exception:
        b = float('nan')
    st.metric("Estimated beta vs IWM (63d)", f"{b:.3f}" if pd.notna(b) else "N/A")

    # Sector exposure
    sec = sector_asof(tp['symbol'].tolist(), ts).fillna("UNKNOWN")
    se = tp.assign(sector=sec.values).groupby('sector')['weight'].sum().sort_values(ascending=False)
    st.subheader("Sector Exposures")
    st.dataframe(se.reset_index().rename(columns={'weight':'sum_weight'}), width='stretch')

    # Concentration
    st.subheader("Concentration")
    st.write("Top 5 names:")
    st.dataframe(tp.head(5), width='stretch')

if __name__ == "__main__":
    app_panel_v16()
