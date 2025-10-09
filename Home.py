"""Streamlit entry point for the Valuation Engine dashboard."""

from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

from modules.valuation_workflow import run_complete_analysis

st.set_page_config(page_title="Valuation Engine", layout="wide")


def _initial_assumptions() -> Dict[str, float]:
    return {
        "risk_free_rate": 0.04,
        "equity_risk_premium": 0.05,
        "target_ebit_margin": 0.18,
        "revenue_growth_rate": [0.12, 0.08],
        "stable_growth_rate": 0.025,
        "sales_to_capital_ratio": 3.0,
        "forecast_period": 10,
    }


def _run_complete_analysis(ticker: str, assumptions: Dict[str, float]) -> None:
    analysis = run_complete_analysis(ticker, assumptions)
    st.session_state["valuation_engine"] = analysis


def _render_overview(state: Dict) -> None:
    st.subheader(f"Latest analysis for {state['ticker']}")
    valuation = state["valuation"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Intrinsic Value / Share", f"${valuation['intrinsic_value_per_share']:.2f}")
    col2.metric("Market Price", f"${valuation['market_price']:.2f}")
    col3.metric("Upside", f"{valuation['upside'] * 100:.1f}%")

    st.markdown("### Overview")
    st.write(
        "Navigate to the dedicated Damodaran, Expectations, and Synthesis pages "
        "using the sidebar to explore assumptions, scenario analysis, and the "
        "variant perception workflow."
    )

    st.markdown("### Quick Facts")
    wacc_summary = state["wacc"]
    st.write(
        {
            "WACC": wacc_summary["wacc"],
            "Cost of Equity": wacc_summary["cost_of_equity"],
            "Cost of Debt": wacc_summary["cost_of_debt"],
            "Beta": wacc_summary["beta"],
            "Market Enterprise Value": state["target_ev"],
            "Market-Implied Growth": state["implied_growth"],
        }
    )


def _sidebar_controls() -> Optional[str]:
    st.sidebar.header("Valuation Controls")
    ticker = st.sidebar.text_input("Ticker", value=st.session_state.get("last_ticker", "SAMPLE"))

    if st.sidebar.button("Run Analysis", use_container_width=True):
        st.session_state["last_ticker"] = ticker
        assumptions = _initial_assumptions()
        try:
            _run_complete_analysis(ticker, assumptions)
            st.sidebar.success("Analysis complete")
        except Exception as exc:  # pragma: no-cover - user feedback path
            st.sidebar.error(str(exc))
    return ticker


_sidebar_controls()

st.title("Valuation Engine Dashboard")

st.markdown(
    """
    This application combines Damodaran's intrinsic valuation methodology with
    Mauboussin's expectations investing framework. Use the sidebar to run a full
    analysis for any supported ticker and then explore the detailed model pages
    from the navigation menu.
    """
)

state = st.session_state.get("valuation_engine")
if state:
    _render_overview(state)
else:
    st.info(
        "Run an analysis from the sidebar to populate the dashboard with "
        "valuation results."
    )
