"""Data access layer for the Valuation Engine dashboard.

This module consolidates all external data retrieval logic and exposes a
provider-agnostic interface that the valuation engines can rely on.  The
implementation follows the resilience-focused design outlined in the technical
specification: every public method attempts to source data from a primary
provider (Polygon via the HTTP API) and automatically falls back to Tiingo or
local sample datasets when required.

The functions return tidy pandas objects that make downstream financial
modelling straightforward.  Any error that prevents the retrieval of data is
surfaced through :class:`DataProviderError` with rich diagnostic information so
that the Streamlit UI can fail gracefully.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional Streamlit caching helpers
# ---------------------------------------------------------------------------
try:  # pragma: no-cover - Streamlit is optional in unit-test environments.
    import streamlit as st

    cache_data = st.cache_data  # type: ignore[attr-defined]
    cache_resource = st.cache_resource  # type: ignore[attr-defined]
except Exception:  # pragma: no-cover - Fallback when Streamlit is unavailable.

    def cache_data(func=None, **_kwargs):
        if func is None:
            return lambda f: f
        return func

    def cache_resource(func=None, **_kwargs):
        if func is None:
            return lambda f: f
        return func


class DataProviderError(RuntimeError):
    """Raised when data for a ticker cannot be retrieved."""


@dataclass
class Quote:
    """Represents a simple equity quote."""

    ticker: str
    price: float
    volume: Optional[int] = None


@dataclass
class CompanyProfile:
    """Minimal subset of descriptive company information."""

    ticker: str
    name: Optional[str]
    industry: Optional[str]
    sector: Optional[str]


SAMPLE_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_financials"
SUPPORTED_STATEMENT_TYPES = {"income_statement", "balance_sheet", "cash_flow"}


class DataProvider:
    """Unified access layer for market and fundamentals data."""

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        polygon_api_key: Optional[str] = None,
        tiingo_api_key: Optional[str] = None,
    ) -> None:
        self._session = session or requests.Session()
        self._polygon_api_key = polygon_api_key or os.getenv("POLYGON_API_KEY")
        self._tiingo_api_key = tiingo_api_key or os.getenv("TIINGO_API_KEY")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_income_statement(
        self, ticker: str, *, period: str = "annual", limit: int = 8
    ) -> pd.DataFrame:
        return self._get_statement_dataframe(ticker, "income_statement", period, limit)

    def get_balance_sheet(
        self, ticker: str, *, period: str = "annual", limit: int = 8
    ) -> pd.DataFrame:
        return self._get_statement_dataframe(ticker, "balance_sheet", period, limit)

    def get_cash_flow_statement(
        self, ticker: str, *, period: str = "annual", limit: int = 8
    ) -> pd.DataFrame:
        return self._get_statement_dataframe(ticker, "cash_flow", period, limit)

    def get_latest_quote(self, ticker: str) -> Quote:
        """Return the latest available price quote for *ticker*."""

        quote = self._fetch_alpaca_quote(ticker)
        if quote:
            return quote

        quote = self._fetch_polygon_quote(ticker)
        if quote:
            return quote

        quote = self._fetch_tiingo_quote(ticker)
        if quote:
            return quote

        quote = self._load_sample_quote(ticker)
        if quote:
            LOGGER.warning("Falling back to sample quote for %s", ticker)
            return quote

        raise DataProviderError(f"Unable to retrieve quote for ticker '{ticker}'")

    def get_company_profile(self, ticker: str) -> CompanyProfile:
        profile = self._fetch_polygon_company_profile(ticker)
        if profile:
            return profile

        profile = self._fetch_tiingo_profile(ticker)
        if profile:
            return profile

        profile = self._load_sample_profile(ticker)
        if profile:
            return profile

        return CompanyProfile(ticker=ticker, name=None, industry=None, sector=None)

    def get_shares_outstanding(self, ticker: str) -> Optional[float]:
        shares = self._fetch_polygon_shares_outstanding(ticker)
        if shares:
            return shares

        shares = self._fetch_tiingo_shares_outstanding(ticker)
        if shares:
            return shares

        sample = self._load_sample_metadata(ticker).get("shares_outstanding")
        if sample:
            return float(sample)
        return None

    def get_beta(self, ticker: str) -> Optional[float]:
        beta = self._fetch_polygon_beta(ticker)
        if beta is not None:
            return beta

        sample = self._load_sample_metadata(ticker).get("beta")
        if sample is not None:
            return float(sample)
        return None

    def get_total_debt(self, ticker: str) -> Optional[float]:
        balance = self.get_balance_sheet(ticker, limit=1)
        if balance.empty:
            return None
        for label in ("TotalDebt", "Total Debt", "total_debt"):
            if label in balance.index:
                return float(balance.loc[label].iloc[0])
        return None

    def get_cash_and_equivalents(self, ticker: str) -> Optional[float]:
        balance = self.get_balance_sheet(ticker, limit=1)
        if balance.empty:
            return None
        for label in (
            "CashAndCashEquivalents",
            "Cash and Cash Equivalents",
            "cash_and_cash_equivalents",
            "Cash",
        ):
            if label in balance.index:
                return float(balance.loc[label].iloc[0])
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_statement_dataframe(
        self, ticker: str, statement_type: str, period: str, limit: int
    ) -> pd.DataFrame:
        if statement_type not in SUPPORTED_STATEMENT_TYPES:
            raise ValueError(f"Unsupported statement type: {statement_type}")

        statements = self._fetch_statements_primary(ticker, statement_type, period, limit)
        if not statements:
            statements = self._fetch_statements_secondary(
                ticker, statement_type, period, limit
            )
        if not statements:
            statements = self._load_sample_statements(
                ticker, statement_type, period, limit
            )
        if not statements:
            LOGGER.error(
                "No financial statements available for ticker %s (%s)",
                ticker,
                statement_type,
            )
            return pd.DataFrame()
        return self._normalize_statements(statements)

    # ----------------------------- Primary providers ------------------
    def _fetch_statements_primary(
        self, ticker: str, statement_type: str, period: str, limit: int
    ) -> List[Dict]:
        if not self._polygon_api_key:
            return []
        try:
            url = "https://api.polygon.io/vX/reference/financials"
            params = {
                "ticker": ticker.upper(),
                "timeframe": "annual" if period == "annual" else "quarterly",
                "limit": limit,
                "order": "desc",
                "include_sources": "true",
            }
            params["apiKey"] = self._polygon_api_key
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # pragma: no-cover - network variability.
            LOGGER.warning("Polygon request failed: %s", exc)
            return []

        results = payload.get("results") or []
        statements: List[Dict] = []
        for entry in results:
            data = entry.get("financials", {})
            normalized = {
                "fiscal_period": entry.get("fiscal_period"),
                "fiscal_year": entry.get("fiscal_year"),
                "data": data.get(statement_type, {}),
            }
            statements.append(normalized)
        return statements

    def _fetch_polygon_quote(self, ticker: str) -> Optional[Quote]:
        if not self._polygon_api_key:
            return None
        try:
            url = "https://api.polygon.io/v2/last/trade/" + ticker.upper()
            params = {"apiKey": self._polygon_api_key}
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            price = payload.get("last", {}).get("price")
            size = payload.get("last", {}).get("size")
            if price is None:
                return None
            return Quote(ticker=ticker.upper(), price=float(price), volume=size)
        except Exception as exc:  # pragma: no-cover - network variability.
            LOGGER.debug("Polygon quote fetch failed: %s", exc)
            return None

    def _fetch_alpaca_quote(self, ticker: str) -> Optional[Quote]:
        api_key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_API_SECRET")
        if not api_key or not secret:
            return None
        try:
            url = f"https://data.alpaca.markets/v2/stocks/{ticker.upper()}/quotes/latest"
            headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret}
            response = self._session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            payload = response.json()
            latest = payload.get("quote") or {}
            price = latest.get("ap") or latest.get("bp") or latest.get("midpoint")
            if price is None:
                return None
            return Quote(
                ticker=ticker.upper(),
                price=float(price),
                volume=latest.get("as") or latest.get("bs"),
            )
        except Exception as exc:  # pragma: no-cover - network variability.
            LOGGER.debug("Alpaca quote fetch failed: %s", exc)
            return None

    def _fetch_polygon_company_profile(self, ticker: str) -> Optional[CompanyProfile]:
        if not self._polygon_api_key:
            return None
        try:
            url = f"https://api.polygon.io/v3/reference/tickers/{ticker.upper()}"
            params = {"apiKey": self._polygon_api_key}
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results") or {}
            return CompanyProfile(
                ticker=ticker.upper(),
                name=results.get("name"),
                industry=results.get("sic_description"),
                sector=results.get("market"),
            )
        except Exception as exc:  # pragma: no-cover
            LOGGER.debug("Polygon profile fetch failed: %s", exc)
            return None

    def _fetch_polygon_shares_outstanding(self, ticker: str) -> Optional[float]:
        if not self._polygon_api_key:
            return None
        try:
            url = "https://api.polygon.io/v3/reference/tickers"
            params = {"ticker": ticker.upper(), "apiKey": self._polygon_api_key}
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results") or []
            if results:
                share_class_shares = results[0].get("share_class_shares_outstanding")
                if share_class_shares is not None:
                    return float(share_class_shares)
        except Exception as exc:  # pragma: no-cover
            LOGGER.debug("Polygon shares outstanding fetch failed: %s", exc)
        return None

    def _fetch_polygon_beta(self, ticker: str) -> Optional[float]:
        if not self._polygon_api_key:
            return None
        try:
            url = f"https://api.polygon.io/v1/meta/symbols/{ticker.upper()}/company"
            params = {"apiKey": self._polygon_api_key}
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            beta = payload.get("beta")
            return float(beta) if beta is not None else None
        except Exception as exc:  # pragma: no-cover
            LOGGER.debug("Polygon beta fetch failed: %s", exc)
            return None

    # ----------------------------- Secondary provider -----------------
    def _fetch_statements_secondary(
        self, ticker: str, statement_type: str, period: str, limit: int
    ) -> List[Dict]:
        if not self._tiingo_api_key:
            return []
        try:
            url = (
                f"https://api.tiingo.com/tiingo/fundamentals/{ticker.upper()}/statements"
            )
            params = {
                "statementType": statement_type,
                "type": "annual" if period == "annual" else "quarterly",
                "limit": limit,
                "format": "json",
                "token": self._tiingo_api_key,
            }
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # pragma: no-cover
            LOGGER.warning("Tiingo request failed: %s", exc)
            return []

        statements: List[Dict] = []
        for entry in payload:
            statements.append(
                {
                    "fiscal_period": entry.get("quarter"),
                    "fiscal_year": entry.get("year"),
                    "data": entry.get("data", {}),
                }
            )
        return statements

    def _fetch_tiingo_quote(self, ticker: str) -> Optional[Quote]:
        if not self._tiingo_api_key:
            return None
        try:
            url = f"https://api.tiingo.com/iex/{ticker.upper()}"
            params = {"token": self._tiingo_api_key}
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list) and payload:
                payload = payload[0]
            price = payload.get("last") or payload.get("mid") or payload.get("tngoLast")
            if price is None:
                return None
            return Quote(
                ticker=ticker.upper(),
                price=float(price),
                volume=payload.get("volume"),
            )
        except Exception as exc:  # pragma: no-cover
            LOGGER.debug("Tiingo quote fetch failed: %s", exc)
            return None

    def _fetch_tiingo_profile(self, ticker: str) -> Optional[CompanyProfile]:
        if not self._tiingo_api_key:
            return None
        try:
            url = f"https://api.tiingo.com/tiingo/fundamentals/{ticker.upper()}/profile"
            params = {"token": self._tiingo_api_key}
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json() or {}
            return CompanyProfile(
                ticker=ticker.upper(),
                name=payload.get("name"),
                industry=payload.get("industry"),
                sector=payload.get("sector"),
            )
        except Exception as exc:  # pragma: no-cover
            LOGGER.debug("Tiingo profile fetch failed: %s", exc)
            return None

    def _fetch_tiingo_shares_outstanding(self, ticker: str) -> Optional[float]:
        if not self._tiingo_api_key:
            return None
        try:
            url = f"https://api.tiingo.com/tiingo/fundamentals/{ticker.upper()}/daily"
            params = {"token": self._tiingo_api_key, "format": "json", "limit": 1}
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            if payload:
                shares = payload[0].get("sharesOutstanding")
                if shares is not None:
                    return float(shares)
        except Exception as exc:  # pragma: no-cover
            LOGGER.debug("Tiingo shares outstanding fetch failed: %s", exc)
            return None
        return None

    # ----------------------------- Sample data fallback ---------------
    def _load_sample_statements(
        self, ticker: str, statement_type: str, period: str, limit: int
    ) -> List[Dict]:
        path = SAMPLE_DATA_PATH / f"{ticker.upper()}_{statement_type}_{period}.json"
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            LOGGER.error("Invalid sample financials at %s: %s", path, exc)
            return []

        statements: List[Dict] = []
        for entry in payload[:limit]:
            statements.append(
                {
                    "fiscal_period": entry.get("fiscal_period"),
                    "fiscal_year": entry.get("fiscal_year"),
                    "data": entry.get("data", {}),
                }
            )
        return statements

    def _load_sample_quote(self, ticker: str) -> Optional[Quote]:
        metadata = self._load_sample_metadata(ticker)
        price = metadata.get("price")
        if price is None:
            return None
        return Quote(ticker=ticker.upper(), price=float(price), volume=metadata.get("volume"))

    def _load_sample_profile(self, ticker: str) -> Optional[CompanyProfile]:
        metadata = self._load_sample_metadata(ticker)
        if not metadata:
            return None
        return CompanyProfile(
            ticker=ticker.upper(),
            name=metadata.get("name"),
            industry=metadata.get("industry"),
            sector=metadata.get("sector"),
        )

    @cache_data(show_spinner=False)
    def _load_sample_metadata(self, ticker: str) -> Dict:
        path = SAMPLE_DATA_PATH / f"{ticker.upper()}_metadata.json"
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError as exc:
            LOGGER.error("Invalid sample metadata at %s: %s", path, exc)
            return {}

    # ----------------------------- Normalisation ----------------------
    def _normalize_statements(self, statements: Iterable[Dict]) -> pd.DataFrame:
        frames: List[pd.Series] = []
        for entry in statements:
            period = entry.get("fiscal_period")
            fiscal_year = entry.get("fiscal_year")
            label = str(fiscal_year)
            if period and period not in {"FY", "Q4", "Q3", "Q2", "Q1"}:
                label = f"{fiscal_year}-{period}"
            data = entry.get("data", {}) or {}
            series = pd.Series(data, name=label, dtype="float64")
            frames.append(series)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, axis=1)
        df = df.sort_index(axis=1)
        df = df.apply(pd.to_numeric, errors="coerce")
        return df


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def create_default_provider() -> DataProvider:
    """Instantiate a :class:`DataProvider` with cached HTTP session."""

    @cache_resource(show_spinner=False)
    def _session_factory() -> requests.Session:
        return requests.Session()

    session = _session_factory()
    return DataProvider(session=session)
