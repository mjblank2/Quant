"""Market expectations analysis view."""

from __future__ import annotations

import streamlit as st

from modules.valuation_workflow import run_complete_analysis

st.set_page_config(page_title="Expectations Investing", layout="wide")

state = st.session_state.get("valuation_engine")
if not state:
    st.warning("Run an analysis from the Home page to populate this view.")
    st.stop()

st.title("Mauboussin Expectations Investing")

implied_tab, strategy_tab = st.tabs(["Implied Expectations", "Strategic Analysis"])

with implied_tab:
    implied_growth = state["implied_growth"]
    expectations = state["expectations"]
    st.metric("Market-implied revenue growth", f"{implied_growth * 100:.2f}%")

    st.markdown("### Benchmark against history")
    comparison = expectations["historical_cagr"]
    st.write({k: f"{v * 100:.2f}%" if v is not None else "N/A" for k, v in comparison.items()})

    st.markdown("### Market-implied pro forma forecast")
    st.dataframe(expectations["pro_forma_forecast"].style.format("{:.0f}"))

with strategy_tab:
    st.subheader("Qualitative assessment")
    prompts = {
        "growth_drivers": "Based on the implied growth rate, what are the most likely sources of growth?",
        "competitive_advantage": "Does the company possess durable advantages to support this performance?",
        "risks_catalysts": "What risks could cause expectations to reset lower and what catalysts could surprise to the upside?",
    }

    for key, label in prompts.items():
        default_value = st.session_state.get(f"expectations_prompt_{key}", "")
        value = st.text_area(label, value=default_value, height=160)
        st.session_state[f"expectations_prompt_{key}"] = value

    if st.button("Refresh analysis with current assumptions"):
        analysis = run_complete_analysis(state["ticker"], state["assumptions"])
        st.session_state["valuation_engine"] = analysis
        st.experimental_rerun()
