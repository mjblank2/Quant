"""
Streamlit page for displaying validation metrics of the 5-day strategy.

This page allows users to compute and view the Sharpe ratio, annualized
Sharpe ratio, probabilistic Sharpe ratio (PSR), and deflated Sharpe ratio
(DSR) for the latest strategy returns stored in the database. Users can
also input a list of Sharpe ratios from candidate strategies to compute
the DSR, which adjusts for multiple testing.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from sqlalchemy import text

from db import engine
from validation.metrics import sharpe_ratio, annualized_sharpe, psr, dsr


def load_returns(limit: int = 365) -> pd.Series:
    """Load recent strategy daily returns from the database."""
    with engine.connect() as con:
        df = pd.read_sql_query(
            text(
                """
                SELECT ts, return
                FROM strategy_daily_returns
                WHERE strategy_name = '5_day_model'
                ORDER BY ts DESC
                LIMIT :limit
                """
            ),
            con,
            params={"limit": limit},
            parse_dates=["ts"],
        )
    if df.empty:
        return pd.Series(dtype=float)
    df = df.sort_values("ts")
    return pd.Series(df["return"].values, index=df["ts"])


def render_validation_page():
    st.set_page_config(page_title="Validation Report", layout="wide")
    st.title("Validation Report — 5-Day Strategy")
    st.caption("Statistical robustness metrics for the 5-day model.")

    returns = load_returns()
    if returns.empty:
        st.warning("No return data available.")
        return
    # Compute metrics
    sr = sharpe_ratio(returns)
    sr_ann = annualized_sharpe(returns)
    psr_val = psr(returns, sr_benchmark=0.0)
    st.metric(label="Periodic Sharpe", value=f"{sr:.4f}")
    st.metric(label="Annualized Sharpe", value=f"{sr_ann:.4f}")
    st.metric(label="Probabilistic Sharpe (PSR)", value=f"{psr_val:.4f}")

    st.subheader("Deflated Sharpe Ratio (DSR)")
    st.write(
        "Enter the Sharpe ratios of all strategies you tested to account for multiple testing."\
        " Example: 0.12, 0.08, 0.15"
    )
    srs_input = st.text_input("Candidate Sharpe Ratios (comma-separated)", value="")
    candidate_srs = []
    if srs_input.strip():
        try:
            candidate_srs = [float(x.strip()) for x in srs_input.split(",")]
        except Exception:
            st.error("Invalid input. Please enter numbers separated by commas.")
    if candidate_srs:
        dsr_val = dsr(returns, candidate_srs)
        st.metric(label="Deflated Sharpe (DSR)", value=f"{dsr_val:.4f}")
    else:
        st.info("DSR will compute after entering candidate Sharpe ratios.")


if __name__ == "__main__" or "streamlit" in __name__:
    render_validation_page()