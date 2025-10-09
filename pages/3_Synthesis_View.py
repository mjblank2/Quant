"""Synthesis view highlighting the expectations gap."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Synthesis View", layout="wide")

state = st.session_state.get("valuation_engine")
if not state:
    st.warning("Run an analysis from the Home page to populate this view.")
    st.stop()

st.title("Synthesis: Variant Perception")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your narrative")
    assumptions = state["assumptions"]
    growth_assumption = assumptions.get("revenue_growth_rate", [0.0])[0]
    st.metric("Assumed revenue growth", f"{growth_assumption * 100:.2f}%")
    st.metric("Target EBIT margin", f"{assumptions.get('target_ebit_margin', 0.0) * 100:.1f}%")
    st.metric("WACC", f"{assumptions.get('wacc', state['wacc']['wacc']) * 100:.2f}%")
    st.write("Valuation narrative")
    st.write(st.session_state.get("valuation_narrative", "No narrative captured yet."))

with col2:
    st.subheader("Market expectations")
    implied_growth = state["implied_growth"]
    st.metric("Implied revenue growth", f"{implied_growth * 100:.2f}%")
    st.metric("Market enterprise value", f"${state['target_ev']:.0f}")
    st.metric("Intrinsic value / share", f"${state['valuation']['intrinsic_value_per_share']:.2f}")

expectations_gap = (growth_assumption - implied_growth) * 100
st.markdown("---")

st.metric("Expectations gap", f"{expectations_gap:.2f} percentage points")

st.markdown(
    """
    The expectations gap quantifies the difference between the growth you
    require in your intrinsic valuation and the growth implied by the market
    price. Use this section to articulate the differentiated insights that make
    up your investment thesis.
    """
)

st.text_area(
    "Variant perception thesis",
    value=st.session_state.get("variant_perception", ""),
    key="variant_perception",
    height=220,
)
