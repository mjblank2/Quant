
from __future__ import annotations

import logging
import os
from datetime import date
from typing import Dict, Any, List
import requests
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from .db import Universe, SessionLocal  # Assumes db.py is in same package

log = logging.getLogger("data.universe")

def _list_small_cap_symbols(max_market_cap: float = 3_000_000_000.0) -> List[Dict[str, Any]]:
    """
    Return a list of symbols and names for all active stocks with a market cap
    below the provided threshold. This function uses Polygon.io's reference
    endpoints to discover tickers and then filters them by market cap.

    Parameters
    ----------
    max_market_cap : float
        The maximum market capitalization (in USD) for a company to be included.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries with 'symbol' and 'name' keys.
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        log.error("POLYGON_API_KEY is not set in the environment.")
        return []

    base_url = "https://api.polygon.io/v3/reference/tickers"
    params = {
        "market": "stocks",
        "active": "true",
        "order": "asc",
        "sort": "ticker",
        "limit": 1000,
        "apiKey": api_key,
    }

    tickers: List[Dict[str, Any]] = []
    next_url: str | None = None
    try:
        while True:
            # If Polygon provides a pagination URL, use it; otherwise, use base_url with params.
            if next_url:
                resp = requests.get(next_url)
            else:
                resp = requests.get(base_url, params=params)
            resp.raise_for_status()
            data = resp.json()

            for result in data.get("results", []):
                symbol = result.get("ticker")
                name = result.get("name")
                # Fetch ticker details to retrieve market_cap
                detail_url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
                detail_resp = requests.get(detail_url, params={"apiKey": api_key})
                if detail_resp.status_code != 200:
                    log.warning("Polygon details request failed for %s: %s", symbol, detail_resp.text)
                    continue
                details = detail_resp.json()
                market_cap = (
                    details.get("results", {}).get("market_cap")
                    if details.get("results") is not None
                    else None
                )
                # Only include if market cap exists and is below threshold
                if market_cap is not None and market_cap < max_market_cap:
                    tickers.append({"symbol": symbol, "name": name})

            next_url = data.get("next_url")
            if not next_url:
                break
    except Exception as e:
        log.error("Error while fetching small cap tickers: %s", e, exc_info=True)
        return []

    return tickers


async def _poly_ticker_info(symbol: str) -> Dict[str, Any]:
    """Fetch ticker info from Polygon (placeholder kept for compatibility)."""
    return {}

async def _poly_adv(symbol: str, start: date, end: date) -> float | None:
    """Fetch average daily volume from Polygon (placeholder kept for compatibility)."""
    return None

def rebuild_universe() -> bool:
    """
    Rebuild the universe table by inserting/updating all symbols with market
    capitalisation below $3B. For each symbol, the function upserts the record
    with included=True in the Universe table.

    Returns
    -------
    bool
        True if the universe was rebuilt successfully, False otherwise.
    """
    log.info("Starting universe rebuild: fetching small cap symbols.")
    symbols = _list_small_cap_symbols()
    if not symbols:
        log.warning("No symbols retrieved for universe rebuild.")
        return False

    try:
        # Insert/Update into the database
        with SessionLocal() as db:
            values = []
            for item in symbols:
                values.append(
                    {
                        "symbol": item["symbol"],
                        "name": item["name"],
                        "included": True,
                    }
                )
            stmt = pg_insert(Universe).values(values)
            update_cols = {
                "name": stmt.excluded.name,
                "included": True,
            }
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=[Universe.symbol],
                set_=update_cols,
            )
            db.execute(on_conflict_stmt)
            db.commit()
        log.info("Universe rebuild completed successfully with %d symbols.", len(symbols))
        return True
    except Exception as e:
        log.error("Failed to rebuild universe: %s", e, exc_info=True)
        return False


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        log.info("Starting universe rebuild...")
        result = rebuild_universe()

        if result is True or result is None:
            log.info("Universe rebuild completed successfully")
            sys.exit(0)
        else:
            log.error("Universe rebuild failed")
            sys.exit(1)

    except Exception as e:
        log.error(f"Universe rebuild failed with exception: {e}", exc_info=True)
        sys.exit(1)
