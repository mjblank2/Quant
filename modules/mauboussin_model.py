"""Implementation of Mauboussin's expectations investing framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional, Sequence

import pandas as pd
from scipy.optimize import brentq

from .damodaran_model import (
    HistoricalData,
    calculate_intrinsic_value,
    forecast_fcff,
)
from .data_provider import DataProvider, DataProviderError


def get_market_enterprise_value(provider: DataProvider, ticker: str) -> float:
    """Compute the market-implied enterprise value."""

    quote = provider.get_latest_quote(ticker)
    shares = provider.get_shares_outstanding(ticker)
    if shares is None:
        raise DataProviderError("Shares outstanding unavailable")
    market_cap = quote.price * shares
    debt = provider.get_total_debt(ticker) or 0.0
    cash = provider.get_cash_and_equivalents(ticker) or 0.0
    enterprise_value = market_cap + debt - cash
    return float(enterprise_value)


def _objective_for_growth(
    growth_rate: float,
    provider: DataProvider,
    ticker: str,
    historical: HistoricalData,
    base_assumptions: Mapping[str, float],
    forecast_period: int,
    target_ev: float,
) -> float:
    assumptions = dict(base_assumptions)
    assumptions["revenue_growth_rate"] = [growth_rate]
    fcff = forecast_fcff(historical, assumptions, forecast_period=forecast_period)
    wacc = assumptions.get("wacc")
    if wacc is None:
        raise ValueError("WACC must be specified in assumptions for reverse DCF")
    valuation = calculate_intrinsic_value(provider, ticker, fcff, wacc, assumptions)
    return valuation["enterprise_value"] - target_ev


def solve_for_implied_growth(
    provider: DataProvider,
    ticker: str,
    historical: HistoricalData,
    target_ev: float,
    base_assumptions: Mapping[str, float],
    *,
    forecast_period: int,
    bracket: Sequence[float] = (-0.2, 1.0),
) -> float:
    """Solve for the revenue growth rate implied by the market price."""

    lower, upper = bracket
    lower = max(lower, -0.95)
    upper = max(upper, lower + 0.05)

    def objective(rate: float) -> float:
        return _objective_for_growth(
            rate, provider, ticker, historical, base_assumptions, forecast_period, target_ev
        )

    f_lower = objective(lower)
    f_upper = objective(upper)
    if f_lower * f_upper > 0:
        # Expand the search interval iteratively
        for step in range(1, 6):
            span = (upper - lower) * (step + 1)
            candidate_upper = upper + span
            f_candidate = objective(candidate_upper)
            if f_lower * f_candidate <= 0:
                upper = candidate_upper
                f_upper = f_candidate
                break
        else:
            raise DataProviderError(
                "Unable to bracket implied growth rate. Adjust assumptions or bracket."
            )

    implied_growth = brentq(objective, lower, upper, maxiter=100)
    return float(implied_growth)


def solve_for_implied_period(
    provider: DataProvider,
    ticker: str,
    historical: HistoricalData,
    target_ev: float,
    base_assumptions: Mapping[str, float],
    *,
    max_years: int = 25,
) -> int:
    """Solve for the number of years growth must persist to justify price."""

    growth_rate = base_assumptions.get("revenue_growth_rate")
    if growth_rate is None:
        raise ValueError("revenue_growth_rate must be provided for period solver")

    def objective(years: float) -> float:
        years_int = int(round(years))
        years_int = max(years_int, 1)
        assumptions = dict(base_assumptions)
        assumptions["revenue_growth_rate"] = [growth_rate]
        fcff = forecast_fcff(historical, assumptions, forecast_period=years_int)
        wacc = assumptions.get("wacc")
        if wacc is None:
            raise ValueError("WACC must be specified in assumptions for reverse DCF")
        valuation = calculate_intrinsic_value(provider, ticker, fcff, wacc, assumptions)
        return valuation["enterprise_value"] - target_ev

    return int(brentq(objective, 1, max_years, maxiter=50))


def _compute_cagr(series: pd.Series, years: int) -> Optional[float]:
    series = series.dropna()
    if len(series) < years + 1:
        return None
    start = series.iloc[-(years + 1)]
    end = series.iloc[-1]
    if start <= 0 or end <= 0:
        return None
    cagr = (end / start) ** (1 / years) - 1
    return float(cagr)


def analyze_expectations(
    provider: DataProvider,
    ticker: str,
    historical: HistoricalData,
    implied_growth_rate: float,
    base_assumptions: Mapping[str, float],
    *,
    forecast_period: int,
) -> Dict[str, object]:
    """Provide context for the implied growth rate."""

    revenue = historical.statements.loc["Revenue"].dropna()
    comparison = {
        "3Y": _compute_cagr(revenue, 3),
        "5Y": _compute_cagr(revenue, 5),
        "10Y": _compute_cagr(revenue, 10),
    }

    assumptions = dict(base_assumptions)
    assumptions["revenue_growth_rate"] = [implied_growth_rate]
    fcff = forecast_fcff(historical, assumptions, forecast_period=forecast_period)

    return {
        "implied_growth_rate": implied_growth_rate,
        "historical_cagr": comparison,
        "pro_forma_forecast": fcff,
    }


__all__ = [
    "analyze_expectations",
    "get_market_enterprise_value",
    "solve_for_implied_growth",
    "solve_for_implied_period",
]
