"""
Financial fundamentals ingestion module.

This module fetches fundamental data from the Polygon API on a per‑symbol basis
and computes a set of point‑in‑time ratios that are useful for quantitative
research.  In addition to the original metrics (debt‑to‑equity, return on
assets, gross margins, profit margins and current ratio), this version
introduces **return on equity (ROE)**.  ROE measures how efficiently a
company generates profit relative to shareholder equity and is a common
fundamental signal.  If either net income or equity is missing or zero,
ROE is recorded as ``None``.
"""

from __future__ import annotations
import asyncio
import pandas as pd
from sqlalchemy import text
from db import engine, upsert_dataframe, Fundamentals
from config import POLYGON_API_KEY
from utils_http import get_json_async
import logging

log = logging.getLogger("data.fundamentals")


def _as_date(x):
    try:
        return pd.to_datetime(x, utc=True).date()
    except Exception:
        return None


async def _poly_financials_async(symbols: list[str], per_symbol_max: int = 12,
                                 batch_size: int = 50) -> pd.DataFrame:
    """Fetch up to N filings per symbol from the Polygon financials endpoint.

    The Polygon API can return large result sets when requesting the entire
    universe.  To keep memory usage bounded, this function processes the
    universe in smaller batches and only keeps one batch of results in
    memory at a time.

    Parameters
    ----------
    symbols : list[str]
        Universe of symbols to fetch fundamentals for.
    per_symbol_max : int, optional
        Maximum filings per symbol, capped at 50 by the API (default 12).
    batch_size : int, optional
        Number of symbols to request concurrently (default 50).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``symbol``, ``as_of``, ``available_at`` and
        fundamental ratios.
    """
    if not POLYGON_API_KEY:
        log.warning("POLYGON_API_KEY not set; skipping fundamentals fetch")
        return pd.DataFrame(columns=['symbol', 'as_of', 'available_at'])
    if not symbols:
        log.info("No symbols provided for fundamentals fetch")
        return pd.DataFrame(columns=['symbol', 'as_of', 'available_at'])

    log.info("Starting fundamentals fetch for %d symbols, has_api_key=%s", len(symbols), bool(POLYGON_API_KEY))

    async def fetch_one(s: str):
        url = "https://api.polygon.io/vX/reference/financials"
        params = {"ticker": s, "limit": min(per_symbol_max, 50), "order": "desc", "apiKey": POLYGON_API_KEY}
        rows = []
        fetched = 0
        next_url = None
        while True:
            js = await get_json_async(next_url or url, params=None if next_url else params)
            if not js or not js.get("results"):
                break
            for res in js["results"]:
                filing_dt = res.get("acceptance_datetime") or res.get("filing_date")
                try:
                    fdt = pd.to_datetime(filing_dt, utc=True)
                    available_at = (fdt + pd.Timedelta(days=1)).normalize().date()
                except Exception:
                    available_at = _as_date(res.get("filing_date"))
                as_of = _as_date(res.get("period_of_report_date")) or _as_date(res.get("fiscal_period_end")) or _as_date(res.get("filing_date"))
                fin = res.get("financials") or {}
                inc = fin.get("income_statement") or {}
                bal = fin.get("balance_sheet") or {}

                revenue = inc.get("revenues") or inc.get("total_revenue")
                gross_profit = inc.get("gross_profit")
                net_income = inc.get("net_income_loss")
                total_assets = bal.get("assets") or bal.get("total_assets")
                equity = bal.get("shareholders_equity") or bal.get("total_shareholders_equity")
                current_assets = bal.get("current_assets")
                current_liab = bal.get("current_liabilities")
                debt = bal.get("long_term_debt") or bal.get("debt") or bal.get("total_debt")

                def _safe_div(a, b):
                    try:
                        a = float(a); b = float(b)
                        if b == 0:
                            return None
                        return a / b
                    except Exception:
                        return None

                gm = _safe_div(gross_profit, revenue)
                pm = _safe_div(net_income, revenue)
                roa = _safe_div(net_income, total_assets)
                cr = _safe_div(current_assets, current_liab)
                de = _safe_div(debt, equity)
                roe = _safe_div(net_income, equity)

                rows.append({
                    "symbol": s,
                    "as_of": as_of,
                    "available_at": available_at,
                    "pe_ttm": None,
                    "pb": None,
                    "ps_ttm": None,
                    "debt_to_equity": de,
                    "return_on_assets": roa,
                    "return_on_equity": roe,
                    "gross_margins": gm,
                    "profit_margins": pm,
                    "current_ratio": cr,
                })
                fetched += 1
                if fetched >= per_symbol_max:
                    break
            next_url = js.get("next_url")
            if next_url and "apiKey" not in next_url:
                next_url = f"{next_url}&apiKey={POLYGON_API_KEY}"
                log.debug("Added authentication to fundamentals pagination URL for %s", s)
            if fetched >= per_symbol_max or not next_url:
                break
        if not rows:
            rows = [{"symbol": s, "as_of": None, "available_at": None}]
        return rows

    all_rows: list[list[dict]] = []
    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i + batch_size]
        batch = await asyncio.gather(*(fetch_one(s) for s in batch_symbols))
        all_rows.extend(batch)
    flat = [r for sub in all_rows for r in (sub if isinstance(sub, list) else [sub])]
    df = pd.DataFrame(flat)
    if not df.empty:
        df = df.dropna(subset=["as_of"])
    return df


def fetch_fundamentals_for_universe(batch_size: int = 50) -> pd.DataFrame:
    """Fetch and upsert financial fundamentals for all symbols in the universe.

    Parameters
    ----------
    batch_size: int, optional
        Number of symbols to request concurrently from Polygon.  Smaller batch
        sizes reduce peak memory usage at the cost of additional HTTP round‑trips.
        Defaults to 50, a safe value for limited‑memory environments.

    Returns
    -------
    pd.DataFrame
        DataFrame of the fetched fundamentals.
    """
    with engine.connect() as con:
        syms = pd.read_sql_query(text("SELECT symbol FROM universe WHERE included = TRUE ORDER BY symbol"), con)["symbol"].tolist()
    if not syms:
        return pd.DataFrame(columns=['symbol', 'as_of', 'available_at'])
    df = asyncio.run(_poly_financials_async(syms, batch_size=batch_size))
    if not df.empty:
        # Upsert into the fundamentals table on (symbol, as_of)
        upsert_dataframe(df, Fundamentals, ["symbol", "as_of"])
    return df


if __name__ == "__main__":
    fetch_fundamentals_for_universe()