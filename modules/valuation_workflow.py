"""Reusable orchestration helpers for the Valuation Engine."""

from __future__ import annotations

from typing import Dict

from modules.data_provider import create_default_provider
from modules.damodaran_model import (
    calculate_intrinsic_value,
    calculate_wacc,
    forecast_fcff,
    get_historical_financials,
)
from modules.mauboussin_model import (
    analyze_expectations,
    get_market_enterprise_value,
    solve_for_implied_growth,
)


def run_complete_analysis(ticker: str, assumptions: Dict[str, float]) -> Dict:
    provider = create_default_provider()
    forecast_period = int(assumptions.get("forecast_period", 10))

    historical = get_historical_financials(provider, ticker)
    wacc_summary = calculate_wacc(provider, ticker, historical, assumptions)

    base_assumptions = dict(assumptions)
    base_assumptions.setdefault("wacc", wacc_summary["wacc"])

    fcff = forecast_fcff(historical, base_assumptions, forecast_period=forecast_period)
    valuation = calculate_intrinsic_value(
        provider, ticker, fcff, wacc_summary["wacc"], base_assumptions
    )

    target_ev = get_market_enterprise_value(provider, ticker)
    implied_growth = solve_for_implied_growth(
        provider,
        ticker,
        historical,
        target_ev,
        base_assumptions,
        forecast_period=forecast_period,
    )
    expectations = analyze_expectations(
        provider,
        ticker,
        historical,
        implied_growth,
        base_assumptions,
        forecast_period=forecast_period,
    )

    return {
        "ticker": ticker.upper(),
        "assumptions": base_assumptions,
        "historical": historical,
        "wacc": wacc_summary,
        "fcff": fcff,
        "valuation": valuation,
        "target_ev": target_ev,
        "implied_growth": implied_growth,
        "expectations": expectations,
        "forecast_period": forecast_period,
    }
