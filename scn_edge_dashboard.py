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
from typing import Optional

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
        " Balance and Disclosure pillars."
    )
    # Create expandable sections for each pillar
    q_values: Dict[str, float] = {}
    b_values: Dict[str, float] = {}
    d_values: Dict[str, float] = {}

    with st.sidebar.expander("Quality (Q)"):
        for name, _weight in QUALITY_SUBFACTORS:
            q_values[name] = st.slider(name, 0.0, 100.0, 50.0, step=1.0)
    with st.sidebar.expander("Balance (B)"):
        for name, _weight in BALANCE_SUBFACTORS:
            b_values[name] = st.slider(name, 0.0, 100.0, 50.0, step=1.0)
    with st.sidebar.expander("Disclosure (D)"):
        for name, _weight in DISCLOSURE_SUBFACTORS:
            d_values[name] = st.slider(name, 0.0, 100.0, 50.0, step=1.0)

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


if __name__ == "__main__":
    main()
