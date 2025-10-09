"""Interactive workspace for Damodaran's intrinsic valuation model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from modules.data_provider import create_default_provider
from modules.damodaran_model import (
    HistoricalData,
    run_scenario_analysis,
    run_sensitivity_analysis,
)
from modules.valuation_workflow import run_complete_analysis

st.set_page_config(page_title="Damodaran Valuation", layout="wide")

state = st.session_state.get("valuation_engine")
if not state:
    st.warning("Run an analysis from the Home page to populate this view.")
    st.stop()

historical: HistoricalData = state["historical"]
assumptions = dict(state["assumptions"])
forecast_period = int(state.get("forecast_period", 10))

st.title("Damodaran Intrinsic Valuation")

assumptions_tab, output_tab, scenario_tab = st.tabs(
    ["Assumptions", "Valuation Output", "Scenario & Sensitivity Analysis"]
)

with assumptions_tab:
    st.subheader("Core Drivers")
    start_growth, end_growth = assumptions.get("revenue_growth_rate", [0.1, 0.06])[:2]
    margin_default = float(assumptions.get("target_ebit_margin", 0.18))
    wacc_default = float(assumptions.get("wacc", state["wacc"]["wacc"]))
    stable_growth_default = float(assumptions.get("stable_growth_rate", 0.025))
    sales_to_capital_default = float(assumptions.get("sales_to_capital_ratio", 3.0))

    with st.form("damodaran-assumptions"):
        col1, col2 = st.columns(2)
        start_growth = col1.slider(
            "Initial revenue growth",
            min_value=0.0,
            max_value=0.40,
            value=float(start_growth),
            step=0.01,
        )
        end_growth = col2.slider(
            "Revenue growth in final forecast year",
            min_value=0.0,
            max_value=0.25,
            value=float(end_growth),
            step=0.005,
        )

        margin = st.slider(
            "Target EBIT margin",
            min_value=0.05,
            max_value=0.45,
            value=margin_default,
            step=0.01,
        )
        wacc = st.slider(
            "Weighted average cost of capital",
            min_value=0.04,
            max_value=0.15,
            value=wacc_default,
            step=0.005,
        )
        stable_growth = st.slider(
            "Terminal (stable) growth rate",
            min_value=0.0,
            max_value=0.05,
            value=stable_growth_default,
            step=0.0025,
        )
        sales_to_capital = st.slider(
            "Sales-to-capital ratio",
            min_value=1.0,
            max_value=8.0,
            value=sales_to_capital_default,
            step=0.1,
        )
        forecast_period = st.slider(
            "Forecast period (years)",
            min_value=3,
            max_value=15,
            value=forecast_period,
        )
        narrative = st.text_area(
            "Valuation narrative",
            value=st.session_state.get("valuation_narrative", ""),
            placeholder="Document the strategic story that justifies these inputs.",
        )

        submitted = st.form_submit_button("Recalculate valuation", use_container_width=True)
        if submitted:
            st.session_state["valuation_narrative"] = narrative
            updated_assumptions = dict(assumptions)
            updated_assumptions.update(
                {
                    "revenue_growth_rate": [start_growth, end_growth],
                    "target_ebit_margin": margin,
                    "wacc": wacc,
                    "stable_growth_rate": stable_growth,
                    "sales_to_capital_ratio": sales_to_capital,
                    "forecast_period": forecast_period,
                }
            )
            analysis = run_complete_analysis(state["ticker"], updated_assumptions)
            st.session_state["valuation_engine"] = analysis
            st.experimental_rerun()

with output_tab:
    st.subheader("Intrinsic Value Summary")
    valuation = state["valuation"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Intrinsic value / share", f"${valuation['intrinsic_value_per_share']:.2f}")
    col2.metric("Market price", f"${valuation['market_price']:.2f}")
    col3.metric("Upside", f"{valuation['upside'] * 100:.1f}%")

    st.markdown("### Projected Cash Flows")
    st.dataframe(state["fcff"].style.format("{:.0f}"))
    st.line_chart(state["fcff"].T[["Revenue", "FCFF"]])

    st.markdown("### Historical Context")
    st.dataframe(historical.statements.tail(10).style.format("{:.0f}"))

with scenario_tab:
    st.subheader("Scenario analysis")
    base_assumptions = dict(state["assumptions"])
    base_wacc = float(base_assumptions.get("wacc", state["wacc"]["wacc"]))
    provider = create_default_provider()

    with st.form("scenario-form"):
        cols = st.columns(3)
        scenario_defs = {}
        for idx, name in enumerate(["Bear", "Base", "Bull"]):
            col = cols[idx]
            col.markdown(f"**{name} case**")
            growth = col.number_input(
                "Growth",
                min_value=-0.10,
                max_value=0.40,
                value=float(base_assumptions.get("revenue_growth_rate", [0.1])[0]),
                key=f"scenario_growth_{name}",
            )
            margin = col.number_input(
                "Margin",
                min_value=0.05,
                max_value=0.45,
                value=float(base_assumptions.get("target_ebit_margin", 0.18)),
                key=f"scenario_margin_{name}",
            )
            wacc_input = col.number_input(
                "WACC",
                min_value=0.04,
                max_value=0.15,
                value=base_wacc,
                key=f"scenario_wacc_{name}",
            )
            scenario_defs[name] = {
                "revenue_growth_rate": [float(growth)],
                "target_ebit_margin": float(margin),
                "wacc": float(wacc_input),
                "stable_growth_rate": base_assumptions.get("stable_growth_rate", 0.025),
                "sales_to_capital_ratio": base_assumptions.get("sales_to_capital_ratio", 3.0),
            }
        run_scenarios = st.form_submit_button("Run scenarios")

    if run_scenarios:
        scenario_df = run_scenario_analysis(
            provider=provider,
            ticker=state["ticker"],
            historical=historical,
            base_wacc=base_wacc,
            scenarios=scenario_defs,
            forecast_period=forecast_period,
        )
        st.session_state["scenario_table"] = scenario_df

    scenario_table = st.session_state.get("scenario_table")
    if isinstance(scenario_table, pd.DataFrame):
        st.dataframe(scenario_table.style.format({"Intrinsic Value": "${:.2f}", "Upside": "{:.1%}"}))

    st.markdown("---")
    st.subheader("Sensitivity analysis")
    parameter = st.selectbox(
        "Parameter",
        options=["wacc", "target_ebit_margin", "revenue_growth_rate"],
    )
    if parameter == "wacc":
        values = np.linspace(0.06, 0.12, 5)
    elif parameter == "target_ebit_margin":
        values = np.linspace(0.12, 0.24, 5)
    else:
        values = np.linspace(0.04, 0.16, 5)

    if st.button("Run sensitivity", key="sensitivity"):
        base_copy = dict(base_assumptions)
        base_copy["forecast_period"] = forecast_period
        sensitivity_df = run_sensitivity_analysis(
            provider=provider,
            ticker=state["ticker"],
            historical=historical,
            base_assumptions=base_copy,
            parameter=parameter,
            values=values,
            forecast_period=forecast_period,
        )
        st.session_state["sensitivity_table"] = sensitivity_df

    sensitivity_table = st.session_state.get("sensitivity_table")
    if isinstance(sensitivity_table, pd.DataFrame):
        st.dataframe(sensitivity_table.style.format("{:.2f}"))
        st.bar_chart(sensitivity_table.set_index(parameter))
