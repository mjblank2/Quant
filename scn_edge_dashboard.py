"""
SCN Edge Score Dashboard
========================

This Streamlit application implements the SCN Edge Score methodology
outlined in the provided Excel template.  It allows an analyst to
research a single stock by entering a ticker symbol and relevant
assumptions, then calculates a normalized edge score ranging from 0 to
10 and assigns a letter grade.  The interface follows the pillars and
factor design of the template, exposing intuitive widgets for each
input and displaying computed outputs in a dashboard‑style layout.

Key features
------------
  * **Automatic price retrieval** – attempts to fetch the latest trade
  price from your configured market data providers (Alpaca, Tiingo or
  Polygon).  The dashboard uses the environment variables
  ``APCA_API_BASE_URL``, ``APCA_API_KEY_ID``, ``APCA_API_SECRET_KEY``,
  ``TIINGO_API_KEY`` and ``POLYGON_API_KEY`` to authenticate
  requests.  If retrieval fails or a manual price override is
  provided, the manual value is used instead.
* **EV normalization** – computes the EV% between the target and
  current price, then applies the tanh‑based compression described in
  the template.
* **Time, confidence and guardrail factors** – translates days to
  catalyst, analyst confidence, quality, balance and disclosure
  subfactors into multiplicative factors.
* **Subfactor weighting** – quality, balance and disclosure are
  decomposed into the same subfactors and weights used in the
  template (e.g. profitability quality, cash runway).  Users can
  assign 0–100 scores per subfactor; the weighted averages are used
  to compute the pillars.
  * **Automatic fundamental scoring** – when a ``POLYGON_API_KEY`` or
    ``TIINGO_API_KEY`` environment variable is available, the
    dashboard attempts to download the latest income statement,
    balance sheet and cash flow information from either Polygon’s
    Financials API or Tiingo’s fundamentals API.  These data are
    parsed to compute core metrics such as profit margin, revenue
    growth, gross margin, leverage, interest coverage, working
    capital health, share dilution and cash runway.  The resulting
    scores pre‑populate the Quality and Balance subfactor sliders.
    If neither API returns data, the sliders default to 50 and can
    be manually adjusted.  Disclosure metrics remain neutral until a
    suitable source is added.
* **Gating and scaling** – applies the hard gate and balance/disclosure
  gate caps before scaling the raw edge score to 0–10 and mapping to
  a letter grade.
* **Watchlist** – calculated results can be appended to a session‑state
  watchlist table for comparison across multiple tickers.

This module should be placed within your Quant repository and
exposed as its own Streamlit page on Render (e.g. by adding a new
entry point or command).  It does not modify any existing files.
"""

from __future__ import annotations

import datetime as _dt
import math as _math
from typing import Dict, List, Tuple

import numpy as _np
import pandas as _pd
import os as _os
from typing import Optional, Any, Callable

# Requests is used to query external market data providers.  If
# requests is unavailable, the import will fail and automatic price
# retrieval will be skipped.
try:
    import requests as _requests  # type: ignore
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False

try:
    import streamlit as st  # type: ignore
except Exception as e:
    raise ImportError(
        "This module requires Streamlit. Install it with `pip install streamlit`"
    ) from e


# -----------------------------------------------------------------------------
# Constants
#
# Subfactor definitions mirror those from the SCN template.  Each entry
# consists of a human‑readable name and its corresponding weight.  The
# weights for each pillar sum to 1.  Feel free to adjust these values
# if your investment process differs from the template.

QUALITY_SUBFACTORS: List[Tuple[str, float]] = [
    ("Profitability quality", 0.25),
    ("Growth durability", 0.20),
    ("Unit economics (GM, LTV/CAC)", 0.20),
    ("Management quality", 0.20),
    ("Competitive advantage (moat)", 0.15),
]

BALANCE_SUBFACTORS: List[Tuple[str, float]] = [
    ("Cash runway (months)", 0.30),
    ("Leverage (net debt/EBITDA)", 0.25),
    ("Interest coverage", 0.20),
    ("Working capital health", 0.15),
    ("Dilution risk (ATM/shelf usage)", 0.10),
]

DISCLOSURE_SUBFACTORS: List[Tuple[str, float]] = [
    ("Filing timeliness", 0.25),
    ("Clarity of KPIs/guidance", 0.25),
    ("Internal controls", 0.20),
    ("Governance quality", 0.20),
    ("IR responsiveness", 0.10),
]

# -----------------------------------------------------------------------------
# Fundamental data helper functions
#
# In order to automatically assign scores for the Quality (Q), Balance (B) and
# Disclosure (D) sub‑factors, we attempt to retrieve fundamental financial
# statements via external APIs.  This implementation leverages the Polygon
# Financials API to pull income statements, balance sheets and cash flow
# statements.  The API is authenticated using the ``POLYGON_API_KEY``
# environment variable.  If data cannot be retrieved or parsed, a neutral
# score of 50 is used for each sub‑factor.  Tiingo and Alpaca are used for
# price retrieval elsewhere but are not currently leveraged for fundamentals.

def _safe_get(d: Any, *keys: str, default: Optional[float] = None) -> Optional[float]:
    """Safely drill into nested dictionaries and return a float.

    Args:
        d: Base dictionary to traverse.
        keys: Sequence of keys to follow.
        default: Value to return if any key is missing or conversion fails.

    Returns:
        Float value if present, else ``default``.
    """
    try:
        for key in keys:
            if isinstance(d, list):
                # if a list is encountered, take the first element
                d = d[0]
            d = d[key]
        if d is None:
            return default
        return float(d)
    except Exception:
        return default


def _fetch_financials_polygon(ticker: str, limit: int = 2) -> Optional[list[dict[str, Any]]]:
    """Fetch financial statements from Polygon.io.

    This helper queries the Polygon Financials endpoint to retrieve a list
    of recent reporting periods for the given ticker.  The ``limit``
    parameter controls how many periods to return (most recent first).  The
    API is accessed via the ``POLYGON_API_KEY`` environment variable.  A
    list of result dictionaries is returned if successful, otherwise
    ``None``.

    The endpoint used is ``https://api.polygon.io/vX/reference/financials``,
    which consolidates income statements, balance sheets and cash flow
    statements derived from SEC filings【918312000758792†L130-L214】.  Each result
    dictionary contains a ``financials`` key with sub‑dictionaries for
    ``income_statement``, ``balance_sheet``, and ``cash_flow_statement``.  See
    Polygon's Financials API glossary for the list of possible fields【918312000758792†L130-L214】.

    Args:
        ticker: The ticker symbol to query (e.g. "AAPL").
        limit: Number of financial periods to retrieve.

    Returns:
        A list of financial period dictionaries, or ``None`` on failure.
    """
    if not (_HAS_REQUESTS and ticker):
        return None
    token = _os.environ.get("POLYGON_API_KEY")
    if not token:
        return None
    url = "https://api.polygon.io/vX/reference/financials"
    params = {
        "ticker": ticker,
        "limit": limit,
        "sort": "reportPeriod",
        "order": "desc",
        "apiKey": token,
    }
    try:
        resp = _requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results")
            if isinstance(results, list) and results:
                return results
    except Exception:
        pass
    return None


def _pick_field(d: dict[str, Any], keys: list[str]) -> Optional[float]:
    """Pick the first available numeric field from a dictionary.

    Many financial statement fields have multiple potential names (e.g. "revenues"
    vs "revenue").  This helper iterates over the provided keys and returns the
    first value that can be coerced to a float.  If no key is found or
    conversion fails, ``None`` is returned.

    Args:
        d: Dictionary to search.
        keys: List of candidate field names in order of preference.

    Returns:
        The numeric value of the first found field or ``None``.
    """
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                continue
    return None


def _compute_metrics_from_polygon(results: list[dict[str, Any]]) -> dict[str, Optional[float]]:
    """Compute fundamental metrics using Polygon financial statements.

    The function expects a list of period dictionaries as returned by
    ``_fetch_financials_polygon``.  The most recent period (index 0) and the
    previous period (index 1) are used to calculate changes such as revenue
    growth.  Metrics computed include profit margin, revenue growth, gross
    margin, leverage, interest coverage, current ratio, share dilution and
    cash runway.  Missing values are handled gracefully and result in
    ``None``.

    Returns:
        Dictionary mapping metric names to numeric values or ``None``.
    """
    metrics: dict[str, Optional[float]] = {
        "profit_margin": None,
        "revenue_growth": None,
        "gross_margin": None,
        "net_debt_to_ebitda": None,
        "interest_coverage": None,
        "current_ratio": None,
        "share_dilution": None,
        "cash_runway_months": None,
    }
    if not results:
        return metrics
    try:
        latest = results[0]
        prev = results[1] if len(results) > 1 else None
        fin_latest = latest.get("financials", {}) if isinstance(latest, dict) else {}
        fin_prev = prev.get("financials", {}) if isinstance(prev, dict) else {}
        inc_latest = fin_latest.get("income_statement", {}) or {}
        bal_latest = fin_latest.get("balance_sheet", {}) or {}
        cf_latest = fin_latest.get("cash_flow_statement", {}) or {}
        inc_prev = fin_prev.get("income_statement", {}) if fin_prev else {}

        # Profit margin: net income / revenue
        revenue_latest = _pick_field(inc_latest, [
            "revenues", "revenue", "total_revenue", "totalRevenues", "salesRevenueNet", "netRevenue"
        ])
        net_income_latest = _pick_field(inc_latest, [
            "net_income_loss_attributable_to_parent", "net_income_loss", "net_income", "netIncome", "netIncomeLoss"
        ])
        if revenue_latest is not None and revenue_latest != 0 and net_income_latest is not None:
            metrics["profit_margin"] = net_income_latest / revenue_latest

        # Revenue growth: (revenue_current - revenue_prev) / revenue_prev
        revenue_prev = None
        if inc_prev:
            revenue_prev = _pick_field(inc_prev, [
                "revenues", "revenue", "total_revenue", "totalRevenues", "salesRevenueNet", "netRevenue"
            ])
            if revenue_latest is not None and revenue_prev and revenue_prev != 0:
                metrics["revenue_growth"] = (revenue_latest - revenue_prev) / revenue_prev

        # Gross margin: gross profit / revenue
        gross_profit = _pick_field(inc_latest, [
            "gross_profit", "grossProfit"
        ])
        if gross_profit is not None and revenue_latest and revenue_latest != 0:
            metrics["gross_margin"] = gross_profit / revenue_latest

        # Leverage: net debt / EBITDA
        liabilities = _pick_field(bal_latest, ["liabilities", "total_liabilities", "totalLiabilities"])
        cash = _pick_field(bal_latest, ["cash", "cash_and_cash_equivalents", "cash_and_equivalents", "cash_and_cash_equivalents_at_carrying_value"])
        # There is no explicit short-term investments field in the glossary; cash proxy is used
        net_debt = None
        if liabilities is not None:
            cash_total = cash or 0.0
            net_debt = max(liabilities - cash_total, 0.0)
        # EBITDA: approximate as operating income + depreciation and amortization
        operating_income = _pick_field(inc_latest, ["operating_income_loss", "operatingIncome", "operating_income"])
        depreciation = _pick_field(inc_latest, ["depreciation_and_amortization", "depreciation_and_amortization_total"])
        ebitda = None
        if operating_income is not None:
            ebitda = operating_income + (depreciation or 0.0)
        if net_debt is not None and ebitda and ebitda != 0:
            metrics["net_debt_to_ebitda"] = net_debt / ebitda

        # Interest coverage: operating income / interest expense
        interest_exp = _pick_field(inc_latest, [
            "interest_expense_operating", "interest_and_debt_expense", "interest_expense"
        ])
        if operating_income is not None and interest_exp and interest_exp != 0:
            # Use absolute value of interest expense (should be positive number)
            metrics["interest_coverage"] = operating_income / abs(interest_exp)

        # Current ratio: current assets / current liabilities
        current_assets = _pick_field(bal_latest, ["current_assets", "total_current_assets", "totalCurrentAssets"])
        current_liabilities = _pick_field(bal_latest, ["current_liabilities", "total_current_liabilities", "totalCurrentLiabilities"])
        if current_assets is not None and current_liabilities and current_liabilities != 0:
            metrics["current_ratio"] = current_assets / current_liabilities

        # Share dilution: difference between diluted and basic shares relative to basic
        diluted_shares = _pick_field(inc_latest, ["diluted_average_shares", "diluted_average_shares_outstanding", "diluted_shares"])
        basic_shares = _pick_field(inc_latest, ["basic_average_shares", "basic_shares"])
        if diluted_shares is not None and basic_shares and basic_shares != 0:
            metrics["share_dilution"] = (diluted_shares - basic_shares) / basic_shares

        # Cash runway: (cash) / (monthly cash burn)
        # Burn rate derived from operating cash flow in cash flow statement
        operating_cf = _pick_field(cf_latest, [
            "net_cash_flow_from_operating_activities", "net_cash_flow_from_operating_activities_continuing", "net_cash_flow_from_operating_activities_discontinued", "net_cash_flow_from_operating_activities_continuing_operations"
        ])
        if operating_cf is not None:
            cash_total = cash or 0.0
            # Positive operating CF means company generates cash; runway infinite
            if operating_cf > 0:
                metrics["cash_runway_months"] = float("inf")
            else:
                monthly_burn = -operating_cf / 12.0
                if monthly_burn > 0:
                    metrics["cash_runway_months"] = cash_total / monthly_burn
    except Exception:
        pass
    return metrics


def _fetch_financials_tiingo(ticker: str, limit: int = 2) -> Optional[list[dict[str, Any]]]:
    """Fetch financial statement data from Tiingo fundamentals.

    Tiingo’s fundamentals API provides quarterly and annual income
    statements, balance sheets and cash flow statements for U.S.
    equities.  This helper queries the statements endpoint to
    retrieve the most recent reporting periods for the specified
    ticker.  The function uses the ``TIINGO_API_KEY`` environment
    variable for authentication.  It constructs a request to
    ``https://api.tiingo.com/tiingo/fundamentals/{ticker}/statements``
    with a limited date range covering the last two years.  If the
    request succeeds, the JSON response (a list of report
    dictionaries) is returned.  Otherwise ``None`` is returned.

    Args:
        ticker: Ticker symbol (e.g. "AAPL").
        limit: Desired number of periods to retrieve (used to slice
            the returned list).  Since the Tiingo API does not
            support an explicit limit parameter, the function
            retrieves all available data in the date window and
            returns only the most recent ``limit`` entries.

    Returns:
        A list of statement dictionaries or ``None`` if the call
        fails or returns no data.
    """
    if not (_HAS_REQUESTS and ticker):
        return None
    token = _os.environ.get("TIINGO_API_KEY")
    if not token:
        return None
    # Define a two‑year date window ending today.  Statements API
    # accepts ISO‑8601 date strings.  Adjust the window length as
    # necessary to capture at least two reporting periods.
    end_date = _dt.date.today().isoformat()
    start_date = (_dt.date.today() - _dt.timedelta(days=365 * 3)).isoformat()
    url = f"https://api.tiingo.com/tiingo/fundamentals/{ticker}/statements"
    params = {
        "startDate": start_date,
        "endDate": end_date,
        "token": token,
        # Request the latest reported data; if you wish to see as‑reported
        # numbers without restatements, set asReported=true.  Here we
        # leave it unspecified to get the most recent values.
    }
    try:
        resp = _requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            # Data is expected to be a list of statement objects ordered
            # by date ascending.  Sort by fiscal end date descending
            # and return the most recent ``limit`` entries.
            if isinstance(data, list) and data:
                # Each entry may include 'statementType' and list of
                # statements, but the structure may vary.  We just
                # return the raw list; the metrics function will
                # interpret as needed.
                # Sort by 'reportDate' or 'calendarDate' if present.
                def get_date(item: Any) -> str:
                    return item.get("reportDate") or item.get("calendarDate") or ""

                sorted_data = sorted(data, key=get_date, reverse=True)
                return sorted_data[:limit]
    except Exception:
        pass
    return None


def _compute_metrics_from_tiingo(results: list[dict[str, Any]]) -> dict[str, Optional[float]]:
    """Compute fundamental metrics using Tiingo financial statements.

    This function expects a list of report dictionaries as returned by
    ``_fetch_financials_tiingo``.  Each report may contain keys such
    as ``cashFlowStatement``, ``incomeStatement`` and
    ``balanceSheet`` (case‑insensitive) holding dictionaries of
    financial values.  Field names differ slightly from Polygon and
    may use camelCase.  The computed metrics mirror those used in
    ``_compute_metrics_from_polygon``.  Missing values are handled
    gracefully.

    Args:
        results: List of statement dicts from Tiingo.

    Returns:
        Dictionary of metric name to value or ``None``.
    """
    metrics: dict[str, Optional[float]] = {
        "profit_margin": None,
        "revenue_growth": None,
        "gross_margin": None,
        "net_debt_to_ebitda": None,
        "interest_coverage": None,
        "current_ratio": None,
        "share_dilution": None,
        "cash_runway_months": None,
    }
    if not results:
        return metrics
    try:
        latest = results[0]
        prev = results[1] if len(results) > 1 else None
        # Unpack statements.  Some keys may be camelCase or snake_case.
        def to_lower_keys(d: dict[str, Any]) -> dict[str, Any]:
            return {k.lower(): v for k, v in d.items()}

        def get_stmt(report: dict[str, Any], name: str) -> dict[str, Any]:
            for key in report.keys():
                if key.lower().startswith(name.lower()):
                    stmt = report.get(key) or {}
                    if isinstance(stmt, dict):
                        return stmt
            return {}

        inc_latest = get_stmt(latest, "incomeStatement")
        bal_latest = get_stmt(latest, "balanceSheet")
        cf_latest = get_stmt(latest, "cashFlowStatement")
        inc_prev = get_stmt(prev, "incomeStatement") if prev else {}

        # Convert keys to lowercase for easier matching
        inc_latest_lower = to_lower_keys(inc_latest)
        bal_latest_lower = to_lower_keys(bal_latest)
        cf_latest_lower = to_lower_keys(cf_latest)
        inc_prev_lower = to_lower_keys(inc_prev)

        # Profit margin: net income / revenue
        revenue_latest = _pick_field(inc_latest_lower, [
            "revenue", "revenues", "total_revenue", "totalrevenues", "netrevenue"
        ])
        net_income_latest = _pick_field(inc_latest_lower, [
            "net_income", "netincomeloss", "net_income_loss", "net_income_loss_attributable_to_parent"
        ])
        if revenue_latest is not None and revenue_latest != 0 and net_income_latest is not None:
            metrics["profit_margin"] = net_income_latest / revenue_latest
        # Revenue growth
        revenue_prev = None
        if inc_prev_lower:
            revenue_prev = _pick_field(inc_prev_lower, [
                "revenue", "revenues", "total_revenue", "totalrevenues", "netrevenue"
            ])
            if revenue_latest is not None and revenue_prev and revenue_prev != 0:
                metrics["revenue_growth"] = (revenue_latest - revenue_prev) / revenue_prev
        # Gross margin
        gross_profit = _pick_field(inc_latest_lower, ["gross_profit", "grossprofit"])
        if gross_profit is not None and revenue_latest and revenue_latest != 0:
            metrics["gross_margin"] = gross_profit / revenue_latest
        # Leverage: net debt / EBITDA
        liabilities = _pick_field(bal_latest_lower, ["total_liabilities", "liabilities"])
        cash = _pick_field(bal_latest_lower, ["cash", "cash_and_cash_equivalents", "cashandequivalents"])
        net_debt = None
        if liabilities is not None:
            cash_total = cash or 0.0
            net_debt = max(liabilities - cash_total, 0.0)
        operating_income = _pick_field(inc_latest_lower, ["operating_income", "operatingincomeloss"])
        depreciation = _pick_field(inc_latest_lower, ["depreciation_and_amortization", "depreciationandamortizationtotal"])
        ebitda = None
        if operating_income is not None:
            ebitda = operating_income + (depreciation or 0.0)
        if net_debt is not None and ebitda and ebitda != 0:
            metrics["net_debt_to_ebitda"] = net_debt / ebitda
        # Interest coverage
        interest_exp = _pick_field(inc_latest_lower, ["interest_expense", "interest_and_debt_expense", "interestexpense"])
        if operating_income is not None and interest_exp and interest_exp != 0:
            metrics["interest_coverage"] = operating_income / abs(interest_exp)
        # Current ratio
        current_assets = _pick_field(bal_latest_lower, ["current_assets", "total_current_assets", "totalcurrentassets"])
        current_liabilities = _pick_field(bal_latest_lower, ["current_liabilities", "total_current_liabilities", "totalcurrentliabilities"])
        if current_assets is not None and current_liabilities and current_liabilities != 0:
            metrics["current_ratio"] = current_assets / current_liabilities
        # Share dilution
        diluted_shares = _pick_field(inc_latest_lower, ["diluted_average_shares", "dilutedaveragesharesoutstanding", "dilutedshares"])
        basic_shares = _pick_field(inc_latest_lower, ["basic_average_shares", "basicshares"])
        if diluted_shares is not None and basic_shares and basic_shares != 0:
            metrics["share_dilution"] = (diluted_shares - basic_shares) / basic_shares
        # Cash runway
        operating_cf = _pick_field(cf_latest_lower, [
            "net_cash_flow_from_operating_activities", "netcashflowfromoperatingactivities"
        ])
        if operating_cf is not None:
            cash_total = cash or 0.0
            if operating_cf > 0:
                metrics["cash_runway_months"] = float("inf")
            else:
                monthly_burn = -operating_cf / 12.0
                if monthly_burn > 0:
                    metrics["cash_runway_months"] = cash_total / monthly_burn
    except Exception:
        pass
    return metrics


def _map_value_to_score(value: Optional[float], thresholds: list[tuple[float, int]], higher_better: bool = True) -> float:
    """Map a numeric value to a 0–100 score based on thresholds.

    Args:
        value: The raw metric value (may be ``None``).
        thresholds: List of (threshold, score) tuples.  For ``higher_better`` metrics
            the list should be sorted in increasing order of threshold; for
            ``higher_better=False`` metrics, thresholds should be sorted in
            decreasing order.  The function returns the score associated with
            the first threshold that the value crosses.
        higher_better: Whether higher values correspond to higher scores.

    Returns:
        A score between 0 and 100.  If ``value`` is ``None``, returns 50.
    """
    if value is None or _math.isnan(value):
        return 50.0
    try:
        val = float(value)
    except Exception:
        return 50.0
    if higher_better:
        for threshold, score in thresholds:
            if val < threshold:
                return float(score)
        return float(thresholds[-1][1])
    else:
        for threshold, score in thresholds:
            if val > threshold:
                return float(score)
        return float(thresholds[-1][1])


def _auto_compute_scores(ticker: str) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """Automatically compute Q/B/D subfactor scores for a given ticker.

    This function attempts to fetch fundamental data and derive meaningful
    subfactor scores.  It returns three dictionaries mapping each subfactor
    name to a score on the 0–100 scale and a dictionary of the raw metrics
    used in the calculations.  If no data could be retrieved, all scores
    default to 50.
    """
    # Initialize all scores to mid‑level values
    q_scores = {name: 50.0 for name, _w in QUALITY_SUBFACTORS}
    b_scores = {name: 50.0 for name, _w in BALANCE_SUBFACTORS}
    d_scores = {name: 50.0 for name, _w in DISCLOSURE_SUBFACTORS}
    raw_metrics: dict[str, float] = {}
    ticker = ticker.strip().upper()
    if not ticker:
        return q_scores, b_scores, d_scores, raw_metrics
    # Retrieve financial statements from Polygon; if unavailable, fall back to Tiingo
    metrics: dict[str, Optional[float]] | None = None
    results_poly = _fetch_financials_polygon(ticker, limit=2)
    if results_poly:
        metrics = _compute_metrics_from_polygon(results_poly)
    else:
        results_tiingo = _fetch_financials_tiingo(ticker, limit=2)
        if results_tiingo:
            metrics = _compute_metrics_from_tiingo(results_tiingo)
    if metrics:
        raw_metrics = metrics
        # Profitability quality -> use profit margin
        q_scores["Profitability quality"] = _map_value_to_score(
            metrics.get("profit_margin"),
            [(-0.1, 10), (0.0, 25), (0.05, 40), (0.1, 60), (0.2, 80), (0.3, 90), (float("inf"), 100)],
            higher_better=True,
        )
        # Growth durability -> use revenue growth
        q_scores["Growth durability"] = _map_value_to_score(
            metrics.get("revenue_growth"),
            [(-0.2, 10), (0.0, 30), (0.05, 50), (0.1, 60), (0.2, 75), (0.4, 90), (float("inf"), 100)],
            higher_better=True,
        )
        # Unit economics (GM, LTV/CAC) -> use gross margin
        q_scores["Unit economics (GM, LTV/CAC)"] = _map_value_to_score(
            metrics.get("gross_margin"),
            [(-0.1, 10), (0.1, 20), (0.2, 40), (0.4, 60), (0.6, 80), (0.8, 90), (float("inf"), 100)],
            higher_better=True,
        )
        # Management quality -> approximate as average of profit margin and revenue growth scores
        q_scores["Management quality"] = (q_scores["Profitability quality"] + q_scores["Growth durability"]) / 2.0
        # Competitive advantage (moat) -> approximate using gross margin score as a proxy
        q_scores["Competitive advantage (moat)"] = q_scores["Unit economics (GM, LTV/CAC)"]
        # Balance metrics
        # Cash runway (months) – longer runway yields higher score
        b_scores["Cash runway (months)"] = _map_value_to_score(
            metrics.get("cash_runway_months"),
            [
                (6.0, 20), (12.0, 40), (18.0, 60), (24.0, 80), (36.0, 90), (float("inf"), 100)
            ],
            higher_better=True,
        )
        # Leverage (net debt/EBITDA)
        b_scores["Leverage (net debt/EBITDA)"] = _map_value_to_score(
            metrics.get("net_debt_to_ebitda"),
            [(0.0, 100), (0.5, 90), (1.0, 80), (2.0, 60), (3.0, 40), (4.0, 20), (float("inf"), 10)],
            higher_better=False,
        )
        # Interest coverage
        b_scores["Interest coverage"] = _map_value_to_score(
            metrics.get("interest_coverage"),
            [(1.0, 10), (2.0, 30), (4.0, 50), (8.0, 70), (12.0, 85), (20.0, 95), (float("inf"), 100)],
            higher_better=True,
        )
        # Working capital health
        b_scores["Working capital health"] = _map_value_to_score(
            metrics.get("current_ratio"),
            [(1.0, 20), (1.5, 40), (2.0, 60), (3.0, 80), (5.0, 90), (float("inf"), 100)],
            higher_better=True,
        )
        # Dilution risk (ATM/shelf usage) -> inverse of share dilution
        dilution = metrics.get("share_dilution")
        if dilution is not None:
            b_scores["Dilution risk (ATM/shelf usage)"] = _map_value_to_score(
                dilution,
                [(-0.2, 100), (-0.1, 90), (0.0, 80), (0.05, 60), (0.1, 40), (0.2, 20), (float("inf"), 10)],
                higher_better=False,
            )
    # Disclosure metrics remain neutral (50) due to lack of automated data
    return q_scores, b_scores, d_scores, raw_metrics

# -----------------------------------------------------------------------------
# Valuation defaults and automatic computation
#
# To remove the need for manual input in the valuation models, we derive
# reasonable default assumptions directly from available financial data.  We
# approximate revenue, shares outstanding, cash and debt from the latest
# financial statements.  Growth and margin assumptions are drawn from the
# automatic fundamental metrics (revenue_growth and profit_margin) when
# available, and fallback constants are used otherwise.  The weighted average
# cost of capital (WACC), tax rate, working‑capital rate, capex rate and
# terminal growth are set to conservative industry‑agnostic values.  These
# defaults feed into the price‑implied expectations and DCF calculations.

def _derive_valuation_defaults(
    ticker: str,
    auto_metrics: Optional[dict[str, Optional[float]]] = None,
) -> Optional[dict[str, float]]:
    """Derive default inputs for valuation models from financial data.

    Args:
        ticker: Stock ticker symbol.
        auto_metrics: Optional dictionary of fundamental metrics returned from
            ``_auto_compute_scores``.  If provided, this dict may contain
            ``profit_margin`` and ``revenue_growth`` entries used to seed
            margin and growth assumptions.

    Returns:
        A dictionary with keys ``sales``, ``shares``, ``cash_excess``,
        ``debt``, ``growth``, ``margin``, ``tax_rate``, ``wc_rate``,
        ``capex_rate``, ``wacc`` and ``terminal_growth``.  If necessary
        information cannot be retrieved, returns ``None``.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        return None
    # Fetch one period of financials to extract revenue, shares, cash and debt
    results = _fetch_financials_polygon(ticker, limit=1)
    using_polygon = True
    if not results:
        results = _fetch_financials_tiingo(ticker, limit=1)
        using_polygon = False
    if not results:
        return None
    sales0 = None
    shares_out = None
    cash_excess = 0.0
    debt_value = 0.0
    try:
        if using_polygon:
            fin = results[0].get("financials", {}) if isinstance(results[0], dict) else {}
            inc = fin.get("income_statement", {}) or {}
            bal = fin.get("balance_sheet", {}) or {}
            sales0 = _pick_field(inc, [
                "revenues", "revenue", "total_revenue", "totalRevenues", "salesRevenueNet", "netRevenue"
            ])
            shares_out = _pick_field(inc, [
                "diluted_average_shares", "diluted_average_shares_outstanding", "diluted_shares"
            ])
            cash_excess = _pick_field(bal, [
                "cash", "cash_and_cash_equivalents", "cash_and_cash_equivalents_at_carrying_value"
            ]) or 0.0
            debt_value = _pick_field(bal, [
                "total_debt", "total_liabilities", "liabilities"
            ]) or 0.0
        else:
            report = results[0] if isinstance(results[0], dict) else {}
            # Extract statements from Tiingo report
            def _extract_stmt_local(report_dict: dict[str, Any], name: str) -> dict[str, Any]:
                for key in report_dict.keys():
                    if key.lower().startswith(name.lower()):
                        stmt = report_dict.get(key) or {}
                        if isinstance(stmt, dict):
                            return stmt
                return {}
            inc_tiingo = _extract_stmt_local(report, "incomeStatement")
            bal_tiingo = _extract_stmt_local(report, "balanceSheet")
            sales0 = _pick_field({k.lower(): v for k, v in inc_tiingo.items()}, [
                "revenue", "revenues", "total_revenue", "totalrevenues", "netrevenue"
            ])
            shares_out = _pick_field({k.lower(): v for k, v in inc_tiingo.items()}, [
                "diluted_average_shares", "dilutedaveragesharesoutstanding", "dilutedshares"
            ])
            cash_excess = _pick_field({k.lower(): v for k, v in bal_tiingo.items()}, [
                "cash", "cash_and_cash_equivalents", "cashandequivalents"
            ]) or 0.0
            debt_value = _pick_field({k.lower(): v for k, v in bal_tiingo.items()}, [
                "total_debt", "total_liabilities", "liabilities", "total_liabilities_net_debt"
            ]) or 0.0
    except Exception:
        return None
    # Derive growth and margin from auto metrics if available
    growth = 0.10  # default 10%
    margin = 0.10  # default 10%
    if auto_metrics:
        rev_growth = auto_metrics.get("revenue_growth")
        prof_margin = auto_metrics.get("profit_margin")
        if rev_growth is not None:
            growth = max(min(float(rev_growth), 0.50), -0.20)  # bound between -20% and +50%
        if prof_margin is not None:
            margin = max(min(float(prof_margin), 0.50), -0.20)  # bound
    # Set conservative default rates
    tax_rate = 0.25
    wc_rate = 0.02
    capex_rate = 0.04
    wacc = 0.10
    terminal_growth = 0.03
    return {
        "sales": float(sales0 or 0.0),
        "shares": float(shares_out or 0.0),
        "cash_excess": float(cash_excess),
        "debt": float(debt_value),
        "growth": float(growth),
        "margin": float(margin),
        "tax_rate": float(tax_rate),
        "wc_rate": float(wc_rate),
        "capex_rate": float(capex_rate),
        "wacc": float(wacc),
        "terminal_growth": float(terminal_growth),
    }


def _auto_compute_valuation(
    ticker: str,
    price_per_share: float,
    auto_metrics: Optional[dict[str, Optional[float]]] = None,
) -> Optional[dict[str, Any]]:
    """Compute expectations and DCF values automatically.

    Args:
        ticker: Ticker symbol.
        price_per_share: Current stock price per share.
        auto_metrics: Optional metrics used to set growth and margin assumptions.

    Returns:
        A dictionary with keys ``implied_years`` and ``dcf_result`` containing the
        market‑implied forecast horizon and DCF output.  If computation fails,
        returns ``None``.
    """
    defaults = _derive_valuation_defaults(ticker, auto_metrics)
    if not defaults:
        return None
    try:
        implied_years = compute_expectations_forecast_period(
            sales0=defaults["sales"],
            price_per_share=price_per_share,
            shares_outstanding=defaults["shares"],
            growth=defaults["growth"],
            margin=defaults["margin"],
            tax_rate=defaults["tax_rate"],
            wc_rate=defaults["wc_rate"],
            capex_rate=defaults["capex_rate"],
            wacc=defaults["wacc"],
            non_operating_assets=defaults["cash_excess"],
            debt=defaults["debt"],
            max_years=20,
        )
        dcf_result = compute_dcf_valuation(
            sales0=defaults["sales"],
            growth=defaults["growth"],
            margin=defaults["margin"],
            tax_rate=defaults["tax_rate"],
            wc_rate=defaults["wc_rate"],
            capex_rate=defaults["capex_rate"],
            wacc=defaults["wacc"],
            terminal_growth=defaults["terminal_growth"],
            non_operating_assets=defaults["cash_excess"],
            debt=defaults["debt"],
            shares_outstanding=defaults["shares"],
            years=5,
        )
        return {
            "implied_years": implied_years,
            "dcf_result": dcf_result,
            "defaults": defaults,
        }
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Expectations Investing and DCF valuation helpers
#
# These functions implement simplified versions of the methods described in
# Michael Mauboussin & Alfred Rappaport's *Expectations Investing* framework
# and Aswath Damodaran's discounted cash‑flow teachings.  They are not
# exhaustive or precise replicas of the authors' models; rather, they provide
# reasonable approximations for educational purposes.

def _project_fcfs(sales0: float, growth: float, margin: float, tax_rate: float,
                  wc_rate: float, capex_rate: float, years: int) -> list[float]:
    """Project free cash flows over a horizon given base sales and assumptions.

    Args:
        sales0: Most recent annual sales (revenue).
        growth: Constant sales growth rate per year (e.g. 0.05 for 5%).
        margin: Operating margin (EBIT/Sales).
        tax_rate: Cash tax rate on EBIT (0–1).
        wc_rate: Incremental working capital as a fraction of the change in sales.
        capex_rate: Incremental fixed capital as a fraction of the change in sales.
        years: Number of years to project.

    Returns:
        List of projected free cash flows for each year 1..years.
    """
    fcfs: list[float] = []
    sales = sales0
    for _ in range(years):
        sales_next = sales * (1.0 + growth)
        # Operating profit and NOPAT
        ebit = sales_next * margin
        nopat = ebit * (1.0 - tax_rate)
        # Incremental investment: change in sales times working‑capital and capex rates
        delta_sales = sales_next - sales
        incremental_investment = delta_sales * (wc_rate + capex_rate)
        fcf = nopat - incremental_investment
        fcfs.append(fcf)
        sales = sales_next
    return fcfs


def _present_value(cash_flows: list[float], discount_rate: float, start_year: int = 1) -> float:
    """Compute the present value of a series of cash flows.

    Args:
        cash_flows: Sequence of cash flows.
        discount_rate: Discount rate (WACC) expressed as a decimal.
        start_year: Year of the first cash flow (default 1).

    Returns:
        Present value of the cash flows.
    """
    pv = 0.0
    for i, cf in enumerate(cash_flows, start=start_year):
        pv += cf / ((1.0 + discount_rate) ** i)
    return pv


def compute_expectations_forecast_period(sales0: float, price_per_share: float, shares_outstanding: float,
                                         growth: float, margin: float, tax_rate: float,
                                         wc_rate: float, capex_rate: float, wacc: float,
                                         non_operating_assets: float, debt: float,
                                         max_years: int = 20) -> int:
    """Estimate the market‑implied forecast period given current price and assumptions.

    This function implements a simplified Price‑Implied Expectations (PIE) analysis.
    It projects free cash flows under the provided assumptions and discounts them
    using WACC.  It gradually extends the forecast horizon until the present
    value of projected FCFs plus non‑operating assets less debt matches or
    exceeds the market value of equity (price × shares).  The function returns
    the number of years required to justify the current price.  If the price
    cannot be justified within ``max_years``, the function returns ``max_years``.

    Args:
        sales0: Latest annual revenue.
        price_per_share: Current share price.
        shares_outstanding: Number of shares outstanding.
        growth: Sales growth rate assumption.
        margin: Operating margin.
        tax_rate: Cash tax rate.
        wc_rate: Working capital investment rate.
        capex_rate: Fixed capital investment rate.
        wacc: Weighted average cost of capital.
        non_operating_assets: Value of excess cash, securities and other non‑operating assets.
        debt: Market value of debt.
        max_years: Maximum horizon to search.

    Returns:
        The implied forecast period in years.
    """
    target_equity_value = price_per_share * shares_outstanding
    enterprise_value_target = target_equity_value + debt - non_operating_assets
    sales = sales0
    fcfs: list[float] = []
    for year in range(1, max_years + 1):
        # Generate one year of free cash flow
        sales_next = sales * (1.0 + growth)
        ebit = sales_next * margin
        nopat = ebit * (1.0 - tax_rate)
        delta_sales = sales_next - sales
        incremental_investment = delta_sales * (wc_rate + capex_rate)
        fcf = nopat - incremental_investment
        fcfs.append(fcf)
        # Compute PV of current list of cash flows
        pv_fcfs = _present_value(fcfs, wacc, start_year=1)
        # Add non‑operating assets (at time zero) and subtract debt to get equity value estimate
        estimated_equity_value = pv_fcfs + non_operating_assets - debt
        if estimated_equity_value >= target_equity_value:
            return year
        sales = sales_next
    return max_years


def compute_dcf_valuation(sales0: float, growth: float, margin: float, tax_rate: float,
                          wc_rate: float, capex_rate: float, wacc: float, terminal_growth: float,
                          non_operating_assets: float, debt: float, shares_outstanding: float,
                          years: int = 5) -> dict[str, float]:
    """Compute an intrinsic share price using a simplified five‑year DCF.

    Args:
        sales0: Latest annual revenue.
        growth: Annual revenue growth rate during the explicit forecast period.
        margin: Operating margin.
        tax_rate: Cash tax rate on EBIT.
        wc_rate: Working capital investment rate.
        capex_rate: Fixed capital investment rate.
        wacc: Weighted average cost of capital.
        terminal_growth: Constant growth rate in perpetuity after the explicit period.
        non_operating_assets: Value of excess cash, securities and other non‑operating assets.
        debt: Market value of debt.
        shares_outstanding: Shares outstanding.
        years: Length of explicit forecast horizon (default 5).

    Returns:
        Dictionary with keys ``enterprise_value``, ``equity_value`` and ``intrinsic_price``.
    """
    # Project FCFF for explicit period
    fcfs = _project_fcfs(sales0, growth, margin, tax_rate, wc_rate, capex_rate, years)
    # PV of explicit cash flows
    pv_fcfs = _present_value(fcfs, wacc, start_year=1)
    # Terminal cash flow (year n+1)
    sales_terminal = sales0 * ((1.0 + growth) ** years) * (1.0 + growth)
    ebit_terminal = sales_terminal * margin
    nopat_terminal = ebit_terminal * (1.0 - tax_rate)
    delta_sales_terminal = sales_terminal - sales0 * ((1.0 + growth) ** years)
    reinvestment_terminal = delta_sales_terminal * (wc_rate + capex_rate)
    fcff_terminal = nopat_terminal - reinvestment_terminal
    # Terminal value using perpetual growth
    terminal_value = fcff_terminal * (1.0 + terminal_growth) / (wacc - terminal_growth)
    # PV of terminal value
    pv_terminal = terminal_value / ((1.0 + wacc) ** years)
    enterprise_value = pv_fcfs + pv_terminal + non_operating_assets - debt
    equity_value = enterprise_value
    intrinsic_price = equity_value / shares_outstanding if shares_outstanding > 0 else float('nan')
    return {
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "intrinsic_price": intrinsic_price,
    }


# -----------------------------------------------------------------------------
# Price retrieval functions
#
# These helper functions attempt to retrieve the latest trade price from
# various market data providers.  They rely on environment variables
# that should be configured in your Render environment.  Each
# function returns ``None`` if the corresponding variables are
# missing, the ``requests`` library is unavailable, or the call
# encounters an error.  See the README or deployment guide for
# expected variable names.

def _fetch_price_alpaca(ticker: str) -> Optional[float]:
    """Fetch the latest trade price using Alpaca's Market Data API.

    The function looks for ``APCA_API_BASE_URL``, ``APCA_API_KEY_ID``
    and ``APCA_API_SECRET_KEY`` in the environment.  It constructs
    a request to the ``/v2/stocks/{symbol}/trades/latest`` endpoint.
    If successful, it returns the trade price (``p``) as a float.
    """
    if not _HAS_REQUESTS:
        return None
    base = _os.environ.get("APCA_API_BASE_URL")
    key_id = _os.environ.get("APCA_API_KEY_ID")
    secret = _os.environ.get("APCA_API_SECRET_KEY")
    if not (base and key_id and secret and ticker):
        return None
    url = f"{base.rstrip('/')}/v2/stocks/{ticker}/trades/latest"
    headers = {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret,
    }
    try:
        resp = _requests.get(url, headers=headers, timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            # Response structure: { "symbol": "TSLA", "trade": { "p": price, ... } }
            trade = data.get("trade") or data.get("trades")
            if isinstance(trade, dict):
                price = trade.get("p") or trade.get("price")
                if price is not None:
                    return float(price)
    except Exception:
        pass
    return None


def _fetch_price_tiingo(ticker: str) -> Optional[float]:
    """Fetch the latest price using Tiingo's IEX endpoint.

    Uses the ``TIINGO_API_KEY`` environment variable.  The API call
    hits ``https://api.tiingo.com/iex/{ticker}`` and returns a list
    containing price fields such as ``last``, ``lastSalePrice`` or
    ``mid``.  This implementation looks for these fields in order
    and returns the first non‑null value.
    """
    if not _HAS_REQUESTS:
        return None
    token = _os.environ.get("TIINGO_API_KEY")
    if not (token and ticker):
        return None
    url = f"https://api.tiingo.com/iex/{ticker}"
    params = {"token": token}
    try:
        resp = _requests.get(url, params=params, timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            # Response is a list of dictionaries
            if isinstance(data, list) and data:
                entry = data[0]
                for field in ("last", "lastSalePrice", "mid", "lastPrice", "lastSize"):
                    price = entry.get(field)
                    if price:
                        try:
                            return float(price)
                        except Exception:
                            continue
                # fallback to close or prevClose
                for field in ("close", "prevClose"):  # End of day values
                    price = entry.get(field)
                    if price:
                        try:
                            return float(price)
                        except Exception:
                            continue
    except Exception:
        pass
    return None


def _fetch_price_polygon(ticker: str) -> Optional[float]:
    """Fetch the latest trade price using Polygon.io.

    Looks for the ``POLYGON_API_KEY`` environment variable.  Calls
    ``https://api.polygon.io/v2/last/trade/{symbol}``.  The response
    contains either ``last.price`` or ``last.p``; this function
    extracts whichever is available.
    """
    if not _HAS_REQUESTS:
        return None
    token = _os.environ.get("POLYGON_API_KEY")
    if not (token and ticker):
        return None
    url = f"https://api.polygon.io/v2/last/trade/{ticker}"
    params = {"apiKey": token}
    try:
        resp = _requests.get(url, params=params, timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            # Response structure: { "symbol": "AAPL", "status": "success", "last": { ... } }
            last = data.get("last") or data.get("results")
            if isinstance(last, dict):
                price = last.get("price") or last.get("p")
                if price is not None:
                    return float(price)
    except Exception:
        pass
    return None


def _fetch_price(ticker: str) -> Optional[float]:
    """Try all configured providers to retrieve a quote.

    The providers are attempted in the following order: Alpaca (if
    ``APCA_API_KEY_ID`` and ``APCA_API_SECRET_KEY`` are set), Tiingo
    (``TIINGO_API_KEY``), and Polygon (``POLYGON_API_KEY``).  The
    first non‑null price is returned.  If none of the providers
    succeed or ``requests`` is unavailable, ``None`` is returned.
    """
    ticker = ticker.strip().upper() if ticker else ""
    if not ticker:
        return None
    price = _fetch_price_alpaca(ticker)
    if price is not None:
        return price
    price = _fetch_price_tiingo(ticker)
    if price is not None:
        return price
    price = _fetch_price_polygon(ticker)
    if price is not None:
        return price
    return None


# Note: The legacy yfinance-based `_fetch_price` function has been
# removed.  Automatic price retrieval now routes through the
# provider-specific helpers defined above.


def _weighted_score(subfactors: List[Tuple[str, float]], values: Dict[str, float]) -> float:
    """Compute a weighted average score for a set of subfactors.

    Args:
        subfactors: List of (name, weight) pairs.
        values: Mapping from name to user–provided 0–100 score.

    Returns:
        Weighted average score in the range [0, 100].  If a value is
        missing for a subfactor, the default is 0.
    """
    total = 0.0
    for name, weight in subfactors:
        total += weight * values.get(name, 0.0)
    # Convert fraction to percentage
    return total * 100.0


def _compute_factors(
    ev_pct: float,
    t_days: int,
    confidence: float,
    q_score: float,
    b_score: float,
    d_score: float,
    forensic_penalty: float,
    hard_gate_fail: bool,
) -> Dict[str, float | int | str]:
    """Compute all intermediate and final factors for the edge score.

    This function encapsulates the math described in the Methodology
    sheet of the SCN template.  It normalizes the EV%, applies time
    decay and confidence adjustments, multiplies by the guardrails and
    then caps the result depending on the gating rules.  Finally it
    scales the score to 0–10 and returns a letter grade.

    Args:
        ev_pct: Expected value percentage difference between target and
            current price.
        t_days: Days until the catalyst date.  Negative values are
            treated as 0.
        confidence: Analyst confidence from 0 to 1.
        q_score: Quality pillar score (0–100).
        b_score: Balance pillar score (0–100).
        d_score: Disclosure pillar score (0–100).
        forensic_penalty: Forensic penalty multiplier (e.g. 1.0,
            0.9, 0.8).
        hard_gate_fail: Whether the hard gate is triggered.

    Returns:
        Dictionary containing all intermediate values and final
        results.
    """
    # Normalise extremes using tanh; base EV% at 10% as neutral
    ev_norm = 50.0 + 50.0 * _math.tanh((ev_pct - 10.0) / 30.0)

    # Days to catalyst; ensure non‑negative
    t_days = max(int(t_days), 0)
    t_factor = 0.7 + 0.3 * _math.exp(-_math.log(2) * t_days / 45.0)

    # Confidence factor; soften penalty for uncertainty
    k_factor = 0.75 + 0.25 * confidence

    # Guardrail multiplication; penalize low quality more severely
    r_guard = (
        (0.5 + 0.5 * q_score / 100.0)
        * (0.6 + 0.4 * b_score / 100.0)
        * (0.7 + 0.3 * d_score / 100.0)
        * forensic_penalty
    )

    # Raw edge score on 0–100 scale
    edge_raw = ev_norm * t_factor * k_factor * r_guard

    # Apply gating caps
    if hard_gate_fail:
        edge_gated = min(edge_raw, 40.0)
    elif b_score < 40.0 or d_score < 40.0:
        edge_gated = min(edge_raw, 60.0)
    else:
        edge_gated = edge_raw

    # Scale to 0–10
    edge = edge_gated / 10.0

    # Letter grade mapping
    if edge >= 8.5:
        grade = "A+"
    elif edge >= 7.5:
        grade = "A"
    elif edge >= 6.5:
        grade = "B"
    elif edge >= 5.0:
        grade = "C"
    else:
        grade = "D"

    return {
        "ev_norm": ev_norm,
        "t_factor": t_factor,
        "k_factor": k_factor,
        "r_guard": r_guard,
        "edge_raw": edge_raw,
        "edge_gated": edge_gated,
        "edge": edge,
        "grade": grade,
        "t_days": t_days,
    }


def _compute_ev_pct(price: float, target: float) -> float:
    """Compute the EV percentage difference between target and price.

    Returns ``0.0`` if either price or target is non‑positive.  The
    value is expressed as a percentage (e.g. 20 means 20%).
    """
    if price <= 0.0 or target <= 0.0:
        return 0.0
    return (target / price - 1.0) * 100.0


def main() -> None:
    """Run the Streamlit dashboard.

    This function defines the layout and user interaction.  It uses
    Streamlit session state to persist the watchlist across runs.
    """
    st.set_page_config(page_title="SCN Edge Score Dashboard", layout="wide")

    st.title("SCN Edge Score Dashboard")
    st.caption(
        "Enter a ticker symbol, your price assumptions and qualitative scores to"
        " generate an SCN edge score and grade.  Use the watchlist to compare"
        " multiple names."
    )

    # Initialize session state for watchlist if absent
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = _pd.DataFrame(
            columns=[
                "Ticker",
                "Price",
                "Target",
                "EV%",
                "Days_to_Catalyst",
                "Confidence",
                "Quality",
                "Balance",
                "Disclosure",
                "Forensic_Penalty",
                "Edge_0_100",
                "Edge_0_10",
                "Grade",
            ]
        )

    # Sidebar input for ticker and price
    st.sidebar.header("Ticker & Price")
    ticker = st.sidebar.text_input(
        "Ticker symbol", value="", max_chars=10, help="Use a valid stock ticker."
    ).upper().strip()

    # Attempt automatic price fetch on blur
    auto_price = None
    if ticker:
        auto_price = _fetch_price(ticker)
    price_default = auto_price if auto_price is not None else 0.0
    manual_price = st.sidebar.number_input(
        "Current price", value=float(price_default), min_value=0.0, step=0.01
    )
    if auto_price is not None:
        st.sidebar.caption(
            f"Automatic price fetched: {auto_price:.2f}. You can override if needed."
        )
    else:
        st.sidebar.caption(
            "Automatic price unavailable – please enter the current price manually."
        )

    target_price = st.sidebar.number_input(
        "Target price", value=0.0, min_value=0.0, step=0.01,
        help="Your price objective for the stock"
    )

    # Catalyst date and confidence
    st.sidebar.header("Catalyst & Confidence")
    today = _dt.date.today()
    catalyst = st.sidebar.date_input(
        "Catalyst date",
        value=today + _dt.timedelta(days=90),
        help="Expected date for a major catalyst"
    )
    confidence = st.sidebar.slider(
        "Confidence (0–1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )

    # Pillar inputs
    st.sidebar.header("Pillar Scores (0–100)")
    st.sidebar.write(
        "Assign scores for each subfactor. The weighted averages form the Quality,"
        " Balance and Disclosure pillars.  If available, the fields below are"
        " pre‑populated with automatically computed scores based on financial data."
    )
    # Compute automatic scores and raw metrics for the selected ticker
    auto_q, auto_b, auto_d, auto_raw = _auto_compute_scores(ticker)

    # Display raw metrics if available
    if auto_raw:
        with st.sidebar.expander("Automated metrics (latest annual report)"):
            for key, val in auto_raw.items():
                if val is None:
                    st.write(f"{key.replace('_', ' ').title()}: N/A")
                else:
                    # Format floats with reasonable precision
                    if abs(val) < 1e6:
                        st.write(f"{key.replace('_', ' ').title()}: {val:.4g}")
                    else:
                        st.write(f"{key.replace('_', ' ').title()}: {val:,.0f}")

    # Create expandable sections for each pillar
    q_values: Dict[str, float] = {}
    b_values: Dict[str, float] = {}
    d_values: Dict[str, float] = {}

    with st.sidebar.expander("Quality (Q)"):
        for name, _weight in QUALITY_SUBFACTORS:
            default_val = float(auto_q.get(name, 50.0))
            q_values[name] = st.slider(name, 0.0, 100.0, default_val, step=1.0)
    with st.sidebar.expander("Balance (B)"):
        for name, _weight in BALANCE_SUBFACTORS:
            default_val = float(auto_b.get(name, 50.0))
            b_values[name] = st.slider(name, 0.0, 100.0, default_val, step=1.0)
    with st.sidebar.expander("Disclosure (D)"):
        for name, _weight in DISCLOSURE_SUBFACTORS:
            default_val = float(auto_d.get(name, 50.0))
            d_values[name] = st.slider(name, 0.0, 100.0, default_val, step=1.0)

    # Compute pillar aggregates
    q_score = _weighted_score(QUALITY_SUBFACTORS, q_values)
    b_score = _weighted_score(BALANCE_SUBFACTORS, b_values)
    d_score = _weighted_score(DISCLOSURE_SUBFACTORS, d_values)

    # Forensic penalty and gating
    st.sidebar.header("Penalty & Gates")
    forensic_penalty = st.sidebar.selectbox(
        "Forensic penalty",
        options=[1.0, 0.9, 0.8],
        index=0,
        format_func=lambda x: f"{x:.1f}" + (" (Normal)" if x == 1.0 else " (Minor/Major)")
    )
    hard_gate_fail = st.sidebar.checkbox(
        "Hard gate failure", value=False,
        help="Check if there is a major disqualifying issue (e.g. fraud)"
    )

    # Compute when user clicks button
    if st.sidebar.button("Calculate Edge Score", disabled=not ticker or manual_price <= 0.0 or target_price <= 0.0):
        ev_pct = _compute_ev_pct(manual_price, target_price)
        t_days = (catalyst - today).days
        factors = _compute_factors(
            ev_pct=ev_pct,
            t_days=t_days,
            confidence=confidence,
            q_score=q_score,
            b_score=b_score,
            d_score=d_score,
            forensic_penalty=float(forensic_penalty),
            hard_gate_fail=hard_gate_fail,
        )
        # Display results in main area
        st.subheader(f"Results for {ticker}")
        colA, colB, colC = st.columns(3)
        colA.metric("EV%", f"{ev_pct:.2f}%")
        colA.metric("EV_norm", f"{factors['ev_norm']:.1f}")
        colB.metric("Days to catalyst", f"{factors['t_days']} d")
        colB.metric("T_factor", f"{factors['t_factor']:.3f}")
        colC.metric("K_factor", f"{factors['k_factor']:.3f}")
        colC.metric("R_guard", f"{factors['r_guard']:.3f}")
        # Raw and gated
        st.write(
            f"**Edge (raw)**: {factors['edge_raw']:.1f} (0–100 scale)\n"
            f"**Edge (gated)**: {factors['edge_gated']:.1f} (0–100 scale)"
        )
        st.subheader(f"Final Score: {factors['edge']:.2f} / 10 ({factors['grade']})")

        # Append to watchlist
        new_row = {
            "Ticker": ticker,
            "Price": manual_price,
            "Target": target_price,
            "EV%": round(ev_pct, 2),
            "Days_to_Catalyst": t_days,
            "Confidence": round(confidence, 2),
            "Quality": round(q_score, 1),
            "Balance": round(b_score, 1),
            "Disclosure": round(d_score, 1),
            "Forensic_Penalty": forensic_penalty,
            "Edge_0_100": round(factors["edge_gated"], 1),
            "Edge_0_10": round(factors["edge"], 2),
            "Grade": factors["grade"],
        }
        st.session_state.watchlist = _pd.concat(
            [st.session_state.watchlist, _pd.DataFrame([new_row])], ignore_index=True
        )


    # Display watchlist
    if not st.session_state.watchlist.empty:
        st.subheader("Watchlist")
        st.dataframe(
            st.session_state.watchlist.sort_values(by="Edge_0_10", ascending=False).reset_index(drop=True),
            hide_index=True,
        )

    # -------------------------------------------------------------------------
    # Valuation Models Section (automatically computed)
    #
    # After the edge score is calculated or whenever a ticker and positive price
    # are provided, we automatically compute a market‑implied forecast period
    # and a five‑year DCF valuation using reasonable default assumptions.
    # These assumptions are derived from the company's most recent financial
    # statements and the automatically computed fundamentals (profit margin and
    # revenue growth).  Users no longer need to manually input values.
    if ticker and manual_price > 0.0:
        valuation_result = _auto_compute_valuation(ticker, manual_price, auto_raw)
        with st.expander("Valuation Models (Expectations Investing & DCF)"):
            st.markdown(
                "This section shows automatically computed valuation analyses. "
                "We derive default assumptions from the latest financial statements "
                "and the auto‑computed fundamentals (growth and margin). The "
                "Price‑Implied Expectations (PIE) analysis infers how many years of "
                "projected cash flows are needed to justify the current stock price. "
                "The five‑year DCF valuation discounts projected free cash flows and "
                "a stable terminal value to estimate intrinsic value per share."
            )
            if valuation_result is None:
                st.warning(
                    "Valuation could not be computed automatically. This may be due "
                    "to missing financial data from Polygon or Tiingo or unavailable "
                    "growth/margin estimates."
                )
            else:
                defaults = valuation_result.get("defaults", {})
                implied_years = valuation_result.get("implied_years")
                dcf_res = valuation_result.get("dcf_result", {})
                # Display assumptions in two columns for readability
                with st.expander("Assumptions used in valuation models"):
                    st.markdown(
                        "**Derived inputs:** These values are extracted from "
                        "the latest financial statements and auto‑computed metrics."
                    )
                    cols = st.columns(3)
                    # Sales and shares
                    cols[0].metric("Current revenue", f"${defaults.get('sales', 0.0):,.0f}")
                    cols[0].metric("Shares outstanding", f"{defaults.get('shares', 0.0):,.0f}")
                    cols[1].metric("Cash/excess", f"${defaults.get('cash_excess', 0.0):,.0f}")
                    cols[1].metric("Debt", f"${defaults.get('debt', 0.0):,.0f}")
                    cols[2].metric("Growth assumption", f"{defaults.get('growth', 0.0)*100:.1f}%")
                    cols[2].metric("Margin assumption", f"{defaults.get('margin', 0.0)*100:.1f}%")
                    cols2 = st.columns(3)
                    cols2[0].metric("WACC", f"{defaults.get('wacc', 0.0)*100:.1f}%")
                    cols2[0].metric("Tax rate", f"{defaults.get('tax_rate', 0.0)*100:.1f}%")
                    cols2[1].metric("Working capital rate", f"{defaults.get('wc_rate', 0.0)*100:.1f}%")
                    cols2[1].metric("Capex rate", f"{defaults.get('capex_rate', 0.0)*100:.1f}%")
                    cols2[2].metric("Terminal growth", f"{defaults.get('terminal_growth', 0.0)*100:.1f}%")
                # Display PIE result
                if implied_years is not None:
                    st.success(f"Market‑implied forecast period: {implied_years} year(s)")
                # Display DCF result
                if dcf_res:
                    st.success(
                        f"DCF enterprise value: ${dcf_res['enterprise_value']:,.0f}\n"
                        f"DCF equity value: ${dcf_res['equity_value']:,.0f}\n"
                        f"Intrinsic share price: ${dcf_res['intrinsic_price']:,.2f}"
                    )


if __name__ == "__main__":
    main()
