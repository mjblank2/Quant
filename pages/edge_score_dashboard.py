"""
SCN Edge Score Dashboard
========================

This Streamlit app implements the full SCN Edge Score methodology as
outlined in the accompanying strategy document. Users enter a stock
symbol and assumptions such as target price, catalyst date,
confidence, and pillar scores. The app fetches a current price from
a free endpoint, computes the expected upside, time decay, confidence
scaling, and pillar guardrails, then applies gating rules before
normalising the result on a 0‑10 scale. Visuals include a gauge for
the final score and a bar chart of the key components.

Place this file into a `pages/` directory within the Streamlit app to
expose it as a new tab. Ensure your environment has `streamlit` and
`plotly` installed and, if necessary, replace the `fetch_price`
function with your preferred data provider.
"""

import datetime
from typing import Optional, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


def fetch_price(ticker: str) -> tuple[Optional[float], Optional[str]]:
    """
    Retrieve the most recent trade price for a given ticker symbol using
    whichever provider credentials are available in the environment.

    The function attempts providers in order of preference: Alpaca,
    Polygon, then Tiingo. If an API key for a provider is not set or
    the request fails, it silently falls back to the next provider.

    The return value is a tuple ``(price, provider)``. ``price`` is the
    latest price as a floating point number, or ``None`` if no provider
    succeeds. ``provider`` is a short string indicating which data
    source was used ("alpaca", "polygon", "tiingo") or ``None`` if
    unavailable. This allows callers to surface the source of the data
    in the UI.

    Note: your Streamlit environment must have network access and the
    corresponding SDKs installed (e.g. ``alpaca_trade_api``). The
    environment should also define the following variables, either via
    ``config.py`` or OS environment variables:

    - ``APCA_API_KEY_ID`` and ``APCA_API_SECRET_KEY`` (for Alpaca)
    - ``APCA_API_BASE_URL`` (defaults to https://paper-api.alpaca.markets)
    - ``POLYGON_API_KEY``
    - ``TIINGO_API_KEY``

    If none are set or all calls fail, the function returns ``(None, None)``.
    """

    # Import API keys from the project config if available. If the import
    # fails (e.g. in a clean environment without the Quant repo), fall
    # back to checking the OS environment variables directly.
    try:
        from config import (
            APCA_API_KEY_ID,
            APCA_API_SECRET_KEY,
            APCA_API_BASE_URL,
            POLYGON_API_KEY,
            TIINGO_API_KEY,
        )  # type: ignore
    except Exception:
        import os
        APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
        APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
        APCA_API_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
        TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")

    ticker = ticker.strip().upper()

    # 1. Alpaca
    if APCA_API_KEY_ID and APCA_API_SECRET_KEY:
        try:
            import alpaca_trade_api as tradeapi  # type: ignore
            api = tradeapi.REST(
                key_id=APCA_API_KEY_ID,
                secret_key=APCA_API_SECRET_KEY,
                base_url=APCA_API_BASE_URL,
            )
            # use get_latest_trade to fetch the last trade price
            latest_trade = api.get_latest_trade(ticker)
            # The return type differs by version; guard accordingly
            price = None
            if hasattr(latest_trade, "price"):
                price = float(latest_trade.price)
            elif isinstance(latest_trade, dict) and "price" in latest_trade:
                price = float(latest_trade["price"])
            if price is not None and price > 0:
                return price, "alpaca"
        except Exception:
            # Log failure silently; will try next provider
            pass

    # 2. Polygon
    if POLYGON_API_KEY:
        try:
            url = f"https://api.polygon.io/v2/last/trade/{ticker}"
            resp = requests.get(url, params={"apiKey": POLYGON_API_KEY}, timeout=10)
            if resp.status_code == 200:
                js = resp.json()
                # expected structure: {"status":"OK","symbol":"AAPL","last":{...,"p":138.94,...}}
                last = js.get("results") or js.get("last") or js.get("results", {})
                # handle both v2 and legacy structures
                if isinstance(last, dict):
                    # polygon v2 uses 'p' for price
                    price = last.get("p") or last.get("price")
                    if price is not None:
                        price_f = float(price)
                        if price_f > 0:
                            return price_f, "polygon"
        except Exception:
            pass

    # 3. Tiingo
    if TIINGO_API_KEY:
        try:
            # Tiingo IEX endpoint returns a list of quotes, each with 'ticker' and 'lastSalePrice'
            url = f"https://api.tiingo.com/iex/{ticker}"
            resp = requests.get(url, params={"token": TIINGO_API_KEY}, timeout=10)
            if resp.status_code == 200:
                js = resp.json()
                if isinstance(js, list) and js:
                    # Some entries may not have the price; search for the ticker
                    for row in js:
                        if (row.get("ticker") or row.get("symbol")).upper() == ticker and row.get("lastSalePrice") is not None:
                            price = float(row["lastSalePrice"])
                            if price > 0:
                                return price, "tiingo"
        except Exception:
            pass

    # No provider succeeded
    return None, None


def compute_edge_score(
    current_price: float,
    target_price: float,
    catalyst_date: datetime.date,
    confidence: float,
    Q: float,
    B: float,
    D: float,
    forensic_penalty: float,
    hard_gate_fail: bool,
    today: Optional[datetime.date] = None,
) -> Dict[str, float]:
    """
    Calculate the SCN Edge Score and its components.
    """
    if today is None:
        today = datetime.date.today()
    if current_price <= 0 or target_price <= 0:
        raise ValueError("Current and target prices must be positive.")
    EV_percent = (target_price / current_price - 1.0) * 100.0
    days_to_catalyst = max(0, (catalyst_date - today).days)
    EV_norm = 50.0 + 50.0 * np.tanh((EV_percent - 10.0) / 30.0)
    T_factor = 0.7 + 0.3 * np.exp(-np.log(2.0) * days_to_catalyst / 45.0)
    K_factor = 0.75 + 0.25 * confidence
    Q_factor = 0.5 + 0.5 * (Q / 100.0)
    B_factor = 0.6 + 0.4 * (B / 100.0)
    D_factor = 0.7 + 0.3 * (D / 100.0)
    R_guard = Q_factor * B_factor * D_factor * forensic_penalty
    edge_raw = float(np.clip(EV_norm * T_factor * K_factor * R_guard, 0.0, 100.0))
    if hard_gate_fail:
        edge_score_capped = min(edge_raw, 40.0)
    elif B < 40.0 or D < 40.0:
        edge_score_capped = min(edge_raw, 60.0)
    else:
        edge_score_capped = edge_raw
    edge_0_10 = edge_score_capped / 10.0
    if edge_score_capped >= 85.0:
        grade = "A+"
    elif edge_score_capped >= 75.0:
        grade = "A"
    elif edge_score_capped >= 65.0:
        grade = "B"
    elif edge_score_capped >= 50.0:
        grade = "C"
    else:
        grade = "D"
    return {
        "EV_percent": EV_percent,
        "days_to_catalyst": days_to_catalyst,
        "EV_norm": EV_norm,
        "T_factor": T_factor,
        "K_factor": K_factor,
        "R_guard": R_guard,
        "edge_raw": edge_raw,
        "edge_score_capped": edge_score_capped,
        "edge_0_10": edge_0_10,
        "grade": grade,
    }


def main() -> None:
    st.set_page_config(page_title="SCN Edge Score", layout="wide")
    st.title("SCN Edge Score Dashboard")
    st.write(
        "The SCN Edge Score combines expected return, catalyst timing, "
        "confidence and fundamental quality into a single rating between 0 and 10."
    )
    with st.form(key="input_form"):
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Ticker symbol", value="AAPL").strip().upper()
            target_price = st.number_input(
                "Target price ($)", min_value=0.01, value=100.0, step=0.01
            )
            catalyst_date = st.date_input(
                "Catalyst date", value=datetime.date.today() + datetime.timedelta(days=30)
            )
        with col2:
            confidence = st.slider(
                "Confidence in thesis", min_value=0.0, max_value=1.0, value=0.75, step=0.01
            )
            Q = st.slider("Quality score (Q)", 0.0, 100.0, 70.0, 1.0)
            B = st.slider("Balance score (B)", 0.0, 100.0, 70.0, 1.0)
            D = st.slider("Disclosure score (D)", 0.0, 100.0, 70.0, 1.0)
        col3, col4 = st.columns(2)
        with col3:
            forensic_penalty = st.slider(
                "Forensic penalty multiplier", 0.50, 1.00, 1.00, 0.01
            )
        with col4:
            hard_gate_fail = st.checkbox("Hard gate fail (deal breaker)", value=False)
        submitted = st.form_submit_button("Calculate Edge Score")

    if submitted:
        if not ticker:
            st.error("Please enter a valid ticker symbol.")
            return
        with st.spinner(f"Fetching current price for {ticker}..."):
            price, provider = fetch_price(ticker)
        if price is None:
            st.error(
                f"Unable to retrieve price for {ticker}. "
                "Please ensure your API keys are set for Alpaca, Polygon or Tiingo, "
                "or update the fetch_price function to use a different provider."
            )
            return
        try:
            results = compute_edge_score(
                current_price=price,
                target_price=target_price,
                catalyst_date=catalyst_date,
                confidence=confidence,
                Q=Q,
                B=B,
                D=D,
                forensic_penalty=forensic_penalty,
                hard_gate_fail=hard_gate_fail,
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        st.subheader(f"Results for {ticker}")
        provider_msg = f"Source: {provider.title()}" if provider else "Source: Unknown"
        st.write(
            f"Current price: **${price:,.2f}** ({provider_msg})\n"
            f"Target price: **${target_price:,.2f}**\n"
            f"Expected upside: **{results['EV_percent']:.1f}%**\n"
            f"Days to catalyst: **{results['days_to_catalyst']}**"
        )
        # Gauge chart
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=results['edge_0_10'],
                number={"suffix": "/10", "valueformat": ".2f"},
                gauge={
                    "axis": {"range": [0, 10]},
                    "bar": {"color": "#1f77b4"},
                    "steps": [
                        {"range": [0, 5], "color": "#f2f2f2"},
                        {"range": [5, 8], "color": "#d9e8fd"},
                        {"range": [8, 10], "color": "#b3d5ff"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": results['edge_0_10'],
                    },
                },
                title={"text": "Edge Score"},
            )
        )
        st.plotly_chart(gauge_fig, use_container_width=True)
        # Bar chart for components
        comp_labels = ["EV_norm", "Time decay", "Confidence", "Guardrail"]
        comp_values = [
            results['EV_norm'],
            results['T_factor'] * 100.0,
            results['K_factor'] * 100.0,
            results['R_guard'] * 100.0,
        ]
        bar_fig = go.Figure(
            data=[
                go.Bar(
                    x=comp_labels,
                    y=comp_values,
                    text=[f"{v:.1f}" for v in comp_values],
                    textposition="outside",
                    marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
                )
            ]
        )
        bar_fig.update_layout(
            title_text="Score Components (0–100 scale)",
            yaxis_title="Component value",
            xaxis_title="Component",
            yaxis=dict(range=[0, 110]),
            showlegend=False,
        )
        st.plotly_chart(bar_fig, use_container_width=True)
        st.success(
            f"**Final Edge Score:** {results['edge_0_10']:.2f} / 10 (Grade {results['grade']})"
        )
        st.write(
            "**Interpretation:**\n"
            "- **8‑10:** High conviction. Strong fundamentals, near catalyst and favourable risk‑reward.\n"
            "- **5‑8:** Moderate edge. Consider improving weaker pillars or refine the thesis.\n"
            "- **Below 5:** Insufficient edge or significant red flags. Proceed cautiously."
        )


if __name__ == "__main__":
    main()

