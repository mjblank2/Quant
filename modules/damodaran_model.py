"""Implementation of Professor Damodaran's intrinsic valuation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data_provider import DataProvider, DataProviderError, Quote, create_default_provider

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


_LINE_ITEM_CANDIDATES: Mapping[str, Tuple[str, ...]] = {
    "Revenue": ("Revenue", "TotalRevenue", "total_revenue", "Sales"),
    "EBIT": ("EBIT", "OperatingIncome", "operating_income"),
    "InterestExpense": ("InterestExpense", "InterestExpenseNonOperating"),
    "IncomeBeforeTax": ("IncomeBeforeTax", "PretaxIncome", "EBT"),
    "IncomeTaxExpense": ("IncomeTaxExpense", "TaxProvision", "Taxes"),
    "NetIncome": ("NetIncome", "NetIncomeCommonStockholders"),
    "WorkingCapital": ("WorkingCapital", "ChangeInWorkingCapital"),
    "TotalDebt": ("TotalDebt", "TotalDebtUSD"),
    "CashAndCashEquivalents": (
        "CashAndCashEquivalents",
        "CashAndShortTermInvestments",
        "Cash",
    ),
    "CapitalExpenditures": ("CapitalExpenditures", "CapitalExpenditure"),
    "DepreciationAndAmortization": (
        "DepreciationAndAmortization",
        "Depreciation",
    ),
}


@dataclass
class HistoricalData:
    """Container for prepared historical financials."""

    statements: pd.DataFrame
    revenue_growth: pd.Series
    operating_margin: pd.Series
    tax_rate: pd.Series
    sales_to_capital: pd.Series


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_line(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for candidate in candidates:
        if candidate in df.index:
            return df.loc[candidate]
    return pd.Series(index=df.columns, dtype="float64")


def get_historical_financials(
    provider: DataProvider, ticker: str, *, num_years: int = 8
) -> HistoricalData:
    """Return a normalised panel of historical statements and derived metrics."""

    income = provider.get_income_statement(ticker, limit=num_years)
    balance = provider.get_balance_sheet(ticker, limit=num_years)
    cash_flow = provider.get_cash_flow_statement(ticker, limit=num_years)

    if income.empty or balance.empty:
        raise DataProviderError(
            f"Insufficient historical financial data for ticker '{ticker}'"
        )

    data: Dict[str, pd.Series] = {}
    for label, candidates in _LINE_ITEM_CANDIDATES.items():
        source = income
        if label in {"TotalDebt", "CashAndCashEquivalents", "WorkingCapital"}:
            source = balance
        elif label in {"CapitalExpenditures", "DepreciationAndAmortization"}:
            source = cash_flow
        data[label] = _extract_line(source, candidates)

    statements = pd.DataFrame(data).T
    statements.columns = income.columns  # use chronological order
    statements = statements.sort_index(axis=1)

    revenue = statements.loc["Revenue"]
    revenue_growth = revenue.pct_change().replace([np.inf, -np.inf], np.nan)

    ebit = statements.loc["EBIT"]
    operating_margin = (ebit / revenue).replace([np.inf, -np.inf], np.nan)

    pretax_income = statements.loc["IncomeBeforeTax"]
    taxes = statements.loc["IncomeTaxExpense"]
    tax_rate = (taxes / pretax_income).clip(lower=0.0, upper=1.0)

    working_capital = statements.loc["WorkingCapital"]
    capex = statements.loc["CapitalExpenditures"]
    depreciation = statements.loc["DepreciationAndAmortization"]
    revenue_delta = revenue.diff()
    invested_capital_delta = (working_capital.diff() + (capex - depreciation).diff())
    sales_to_capital = revenue_delta / invested_capital_delta.replace(0, np.nan)

    derived = {
        "Revenue Growth Rate": revenue_growth,
        "Operating Margin": operating_margin,
        "Effective Tax Rate": tax_rate,
        "Sales to Capital Ratio": sales_to_capital,
    }
    derived_df = pd.DataFrame(derived).T

    combined = pd.concat([statements, derived_df])
    combined = combined.sort_index(axis=1)

    return HistoricalData(
        statements=combined,
        revenue_growth=revenue_growth,
        operating_margin=operating_margin,
        tax_rate=tax_rate,
        sales_to_capital=sales_to_capital,
    )


# ---------------------------------------------------------------------------
# Valuation logic
# ---------------------------------------------------------------------------


def calculate_wacc(
    provider: DataProvider,
    ticker: str,
    historical: HistoricalData,
    user_assumptions: MutableMapping[str, float],
) -> Dict[str, float]:
    """Calculate the Weighted Average Cost of Capital (WACC)."""

    assumptions = {
        "risk_free_rate": 0.04,
        "equity_risk_premium": 0.05,
        **user_assumptions,
    }

    beta = assumptions.get("beta")
    if beta is None:
        beta = provider.get_beta(ticker) or 1.0
    risk_free_rate = assumptions.get("risk_free_rate", 0.04)
    equity_risk_premium = assumptions.get("equity_risk_premium", 0.05)

    cost_of_equity = risk_free_rate + beta * equity_risk_premium

    interest_expense = historical.statements.loc["InterestExpense"].dropna().iloc[-1]
    total_debt = historical.statements.loc["TotalDebt"].dropna().iloc[-1]
    pretax_cost_of_debt = (interest_expense / total_debt) if total_debt else 0.0

    effective_tax_rate = (
        assumptions.get("tax_rate")
        or historical.tax_rate.dropna().iloc[-1]
        if not historical.tax_rate.dropna().empty
        else 0.25
    )
    cost_of_debt = pretax_cost_of_debt * (1 - effective_tax_rate)

    quote = provider.get_latest_quote(ticker)
    shares = provider.get_shares_outstanding(ticker) or 0.0
    market_cap = quote.price * shares

    market_value_debt = total_debt
    total_capital = market_cap + market_value_debt
    if total_capital == 0:
        raise DataProviderError("Unable to compute capital structure weights")

    weight_equity = market_cap / total_capital
    weight_debt = market_value_debt / total_capital

    wacc = weight_equity * cost_of_equity + weight_debt * cost_of_debt

    return {
        "beta": beta,
        "risk_free_rate": risk_free_rate,
        "equity_risk_premium": equity_risk_premium,
        "cost_of_equity": cost_of_equity,
        "cost_of_debt": cost_of_debt,
        "wacc": wacc,
        "market_cap": market_cap,
        "market_value_debt": market_value_debt,
        "weight_equity": weight_equity,
        "weight_debt": weight_debt,
    }


def _prepare_growth_vector(
    growth_input: Sequence[float], forecast_period: int
) -> np.ndarray:
    if len(growth_input) == forecast_period:
        return np.array(growth_input, dtype=float)
    if len(growth_input) == 2:
        return np.linspace(growth_input[0], growth_input[1], forecast_period)
    return np.repeat(growth_input[0], forecast_period)


def _prepare_margin_vector(
    current_margin: float, target_margin: float, forecast_period: int
) -> np.ndarray:
    return np.linspace(current_margin, target_margin, forecast_period)


def forecast_fcff(
    historical: HistoricalData,
    user_assumptions: Mapping[str, float],
    *,
    forecast_period: int,
) -> pd.DataFrame:
    """Project FCFF based on revenue growth, margins and reinvestment efficiency."""

    revenue_history = historical.statements.loc["Revenue"].dropna()
    last_revenue = revenue_history.iloc[-1]

    margin_history = historical.operating_margin.dropna()
    current_margin = margin_history.iloc[-1] if not margin_history.empty else 0.15

    sales_to_capital_history = historical.sales_to_capital.dropna()
    base_sales_to_capital = (
        sales_to_capital_history.iloc[-1] if not sales_to_capital_history.empty else 3.0
    )

    tax_rate_history = historical.tax_rate.dropna()
    tax_rate = user_assumptions.get("tax_rate") or (
        tax_rate_history.iloc[-1] if not tax_rate_history.empty else 0.25
    )

    growth_input = user_assumptions.get("revenue_growth_rate", [0.08])
    if isinstance(growth_input, (int, float)):
        growth_input = [float(growth_input)]
    elif isinstance(growth_input, tuple):
        growth_input = list(growth_input)
    growth_vector = _prepare_growth_vector(growth_input, forecast_period)

    target_margin = user_assumptions.get("target_ebit_margin", current_margin)
    margin_vector = _prepare_margin_vector(current_margin, target_margin, forecast_period)

    sales_to_capital = user_assumptions.get(
        "sales_to_capital_ratio", base_sales_to_capital
    )

    columns: List[str] = []
    rows = {
        "Revenue": [],
        "EBIT": [],
        "EBIT(1-T)": [],
        "Reinvestment": [],
        "FCFF": [],
    }

    revenue = last_revenue
    last_year = revenue_history.index[-1]
    try:
        start_year = int(str(last_year))
    except ValueError:
        start_year = pd.Timestamp.today().year

    for step in range(forecast_period):
        growth = growth_vector[step]
        revenue *= 1 + growth
        margin = margin_vector[step]
        ebit = revenue * margin
        nopat = ebit * (1 - tax_rate)
        previous_revenue = rows["Revenue"][-1] if rows["Revenue"] else last_revenue
        reinvestment = 0.0
        if sales_to_capital:
            reinvestment = max((revenue - previous_revenue) / sales_to_capital, 0.0)
        fcff = nopat - reinvestment

        year_label = str(start_year + step + 1)
        columns.append(year_label)
        rows["Revenue"].append(revenue)
        rows["EBIT"].append(ebit)
        rows["EBIT(1-T)"].append(nopat)
        rows["Reinvestment"].append(reinvestment)
        rows["FCFF"].append(fcff)

    forecast_df = pd.DataFrame(rows, index=rows.keys(), columns=columns)
    return forecast_df


def calculate_intrinsic_value(
    provider: DataProvider,
    ticker: str,
    fcff_forecast: pd.DataFrame,
    wacc: float,
    user_assumptions: Mapping[str, float],
) -> Dict[str, float]:
    """Discount projected FCFFs and compute the intrinsic value per share."""

    stable_growth_rate = user_assumptions.get("stable_growth_rate", 0.025)
    if stable_growth_rate >= wacc:
        raise ValueError("Stable growth rate must be less than WACC")

    discount_rate = wacc
    years = np.arange(1, fcff_forecast.shape[1] + 1)
    fcff_values = fcff_forecast.loc["FCFF"].values
    discounted_fcffs = fcff_values / np.power(1 + discount_rate, years)

    terminal_fcff = fcff_values[-1] * (1 + stable_growth_rate)
    terminal_value = terminal_fcff / (discount_rate - stable_growth_rate)
    terminal_discount_factor = (1 + discount_rate) ** years[-1]
    discounted_terminal_value = terminal_value / terminal_discount_factor

    enterprise_value = discounted_fcffs.sum() + discounted_terminal_value

    cash = provider.get_cash_and_equivalents(ticker) or 0.0
    debt = provider.get_total_debt(ticker) or 0.0
    equity_value = enterprise_value - debt + cash

    shares = provider.get_shares_outstanding(ticker) or 0.0
    if shares == 0:
        raise DataProviderError("Shares outstanding unavailable")

    intrinsic_value_per_share = equity_value / shares

    market_price = provider.get_latest_quote(ticker).price
    upside = (intrinsic_value_per_share / market_price) - 1 if market_price else np.nan

    return {
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "intrinsic_value_per_share": intrinsic_value_per_share,
        "terminal_value": terminal_value,
        "discounted_terminal_value": discounted_terminal_value,
        "pv_of_fcff": discounted_fcffs.sum(),
        "upside": upside,
        "market_price": market_price,
    }


# ---------------------------------------------------------------------------
# Scenario and sensitivity utilities
# ---------------------------------------------------------------------------


def run_scenario_analysis(
    provider: DataProvider,
    ticker: str,
    historical: HistoricalData,
    base_wacc: float,
    scenarios: Mapping[str, Mapping[str, float]],
    forecast_period: int,
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for name, assumptions in scenarios.items():
        fcff = forecast_fcff(historical, assumptions, forecast_period=forecast_period)
        summary = calculate_intrinsic_value(
            provider,
            ticker,
            fcff,
            assumptions.get("wacc", base_wacc),
            assumptions,
        )
        growth_value = assumptions.get("revenue_growth_rate")
        if isinstance(growth_value, (list, tuple)):
            growth_display = growth_value[0]
        else:
            growth_display = growth_value
        record = {
            "Scenario": name,
            "Intrinsic Value": summary["intrinsic_value_per_share"],
            "Upside": summary["upside"],
            "WACC": assumptions.get("wacc", base_wacc),
            "Revenue Growth": growth_display,
            "Margin": assumptions.get("target_ebit_margin"),
        }
        records.append(record)
    return pd.DataFrame(records)


def run_sensitivity_analysis(
    provider: DataProvider,
    ticker: str,
    historical: HistoricalData,
    base_assumptions: Mapping[str, float],
    parameter: str,
    values: Sequence[float],
    forecast_period: int,
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for value in values:
        assumptions = dict(base_assumptions)
        assumptions[parameter] = value
        fcff = forecast_fcff(historical, assumptions, forecast_period=forecast_period)
        wacc = assumptions.get("wacc") or base_assumptions.get("wacc") or 0.08
        summary = calculate_intrinsic_value(provider, ticker, fcff, wacc, assumptions)
        records.append(
            {
                parameter: value,
                "Intrinsic Value": summary["intrinsic_value_per_share"],
            }
        )
    return pd.DataFrame(records)


__all__ = [
    "HistoricalData",
    "calculate_intrinsic_value",
    "calculate_wacc",
    "forecast_fcff",
    "get_historical_financials",
    "run_scenario_analysis",
    "run_sensitivity_analysis",
]
