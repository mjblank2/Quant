
from __future__ import annotations

import logging
import os
from datetime import date
from typing import Dict, Any, List
import requests
from sqlalchemy.dialects.postgresql import insert as pg_insert
from .db import Universe, SessionLocal  # Assumes db.py is in same package

# Import robust HTTP utility for reliable API calls
try:
    from utils_http import get_json
    HAS_UTILS_HTTP = True
except ImportError:
    HAS_UTILS_HTTP = False

log = logging.getLogger("data.universe")


def _get_polygon_api_key() -> str:
    """
    Get and validate the Polygon API key from environment.

    Returns
    -------
    str
        The API key if valid

    Raises
    ------
    RuntimeError
        If POLYGON_API_KEY is not set or empty
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "POLYGON_API_KEY environment variable is required but not set. "
            "Please set your Polygon.io API key in the environment "
            "(e.g., in Render environment variables)."
        )
    return api_key.strip()


def _robust_get_json(url: str, params: Dict[str, Any] = None, timeout: float = 60.0) -> Dict[str, Any]:
    """
    Robust HTTP GET with retry logic and timeout handling.

    Uses utils_http.get_json if available (with retry logic), falls back to requests.get.
    Increased default timeout for large paginated responses.

    Parameters
    ----------
    url : str
        URL to fetch
    params : Dict[str, Any], optional
        Query parameters
    timeout : float, optional
        Request timeout in seconds (default: 60s for large responses)

    Returns
    -------
    Dict[str, Any]
        JSON response data, empty dict on error
    """
    if HAS_UTILS_HTTP:
        # Use robust HTTP utility with retry logic and structured logging
        return get_json(url, params=params, timeout=timeout, max_tries=3)
    else:
        # Fallback to basic requests with improved timeout
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning("HTTP request failed for %s: %s", url, e)
            return {}


def _add_polygon_auth(url: str, api_key: str) -> str:
    """
    Ensure a Polygon.io URL includes authentication.

    Parameters
    ----------
    url : str
        The URL to authenticate
    api_key : str
        The API key to use

    Returns
    -------
    str
        URL with API key parameter added if not already present
    """
    if "apiKey=" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}apiKey={api_key}"
    return url


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
    # Validate API key early and fail fast if missing
    try:
        api_key = _get_polygon_api_key()
        has_api_key = True
    except RuntimeError as e:
        log.error(str(e))
        return []

    log.info("Starting small cap ticker fetch, has_api_key=%s", has_api_key)

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
            # If Polygon provides a pagination URL, ensure it has auth; otherwise, use base_url with params.
            if next_url:
                # Ensure pagination URL includes authentication
                authenticated_url = _add_polygon_auth(next_url, api_key)
                log.info("Making paginated request with cursor, has_auth=%s", "apiKey=" in authenticated_url)
                data = _robust_get_json(authenticated_url, timeout=60.0)
            else:
                log.info("Making initial request, has_auth=%s", "apiKey" in params)
                data = _robust_get_json(base_url, params=params, timeout=60.0)

            if not data:
                log.error("Failed to fetch data from Polygon API")
                break

            for result in data.get("results", []):
                symbol = result.get("ticker")
                name = result.get("name")

                # Skip if basic info is missing
                if not symbol or not name:
                    continue

                # Fetch ticker details to retrieve market_cap using robust HTTP
                detail_url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
                details = _robust_get_json(detail_url, params={"apiKey": api_key}, timeout=60.0)

                if not details:
                    log.warning("Polygon details request failed for %s", symbol)
                    continue

                market_cap = (
                    details.get("results", {}).get("market_cap")
                    if details.get("results") is not None
                    else None
                )
                # Only include if market cap exists and is below threshold
                if market_cap is not None and market_cap < max_market_cap:
                    tickers.append({"symbol": symbol, "name": name})
                    log.debug("Added %s (market_cap: $%.2fB)", symbol, market_cap / 1_000_000_000)

                # Log progress every 100 symbols processed for long-running operations
                if len(data.get("results", [])) > 50 and len(tickers) % 50 == 0:
                    log.info("Progress: processed %d tickers, found %d small-cap symbols so far",
                             len(data.get("results", [])), len(tickers))

            next_url = data.get("next_url")
            if not next_url:
                break

    except Exception as e:
        # Since we're using robust HTTP utility, most HTTP errors are already handled
        # This catches any remaining exceptions like JSON parsing or logic errors
        log.error("Error while fetching small cap tickers: %s", e, exc_info=True)
        return []

    log.info("Completed small cap fetch: found %d symbols total", len(tickers))
    return tickers


def test_polygon_api_connection() -> bool:
    """
    Simple smoke test to verify Polygon API connectivity and authentication.

    Returns
    -------
    bool
        True if the API connection works, False otherwise
    """
    try:
        api_key = _get_polygon_api_key()
        log.info("Testing Polygon API connection, has_api_key=%s", bool(api_key))

        # Make a minimal request to test connectivity using robust HTTP
        base_url = "https://api.polygon.io/v3/reference/tickers"
        params = {
            "market": "stocks",
            "active": "true",
            "limit": 1,
            "apiKey": api_key,
        }

        data = _robust_get_json(base_url, params=params, timeout=30.0)

        if not data:
            log.error("Polygon API test failed: no data received")
            return False

        results_count = len(data.get("results", []))
        log.info("Polygon API test successful, received %d results", results_count)
        return True

    except RuntimeError as e:
        log.warning("Polygon API test skipped: %s", e)
        return False
    except Exception as e:
        log.error("Polygon API test failed: %s", e)
        return False


def _poly_ticker_info() -> Dict[str, Any]:
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
        # Test API connection first
        log.info("Testing Polygon API connection...")
        if test_polygon_api_connection():
            log.info("API connection test passed")
        else:
            log.warning("API connection test failed - proceeding anyway")

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
