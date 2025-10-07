from __future__ import annotations

import logging
import os
import urllib.parse
import time
from datetime import date, datetime, timezone
from typing import Dict, Any, List

import requests

from .db import Universe  # Assumes db.py is in same package

try:  # pragma: no cover - convenience for tests
    from db import SessionLocal as SessionLocal  # type: ignore
except Exception:  # pragma: no cover - fallback when db module unavailable
    SessionLocal = None

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

def _safe_get_json(
    url: str,
    params: dict | None = None,
    timeout: int = 30,
    retries: int = 3,
    backoff_factor: float = 0.5,
) -> dict | None:
    """
    Make a safe HTTP GET request that returns JSON or None on failure.

    Parameters
    ----------
    url : str
        The URL to request
    params : dict, optional
        URL parameters to include
    timeout : int, default 30
        Request timeout in seconds

    Returns
    -------
    dict or None
        The JSON response if successful, None if any error occurs
    """
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        # Log the actual request URL (resp.url includes params)
        actual_url = getattr(resp, 'url', url) if 'resp' in locals() else url
        log.warning("Request failed for %s: %s", actual_url, str(e))
        return None
    attempt = 0
    while attempt <= retries:
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)

            if status_code is not None and 500 <= status_code < 600 and attempt < retries:
                sleep_seconds = backoff_factor * (2**attempt)
                log.warning(
                    "Request failed for %s with status %s, retrying in %.1fs (attempt %d/%d)",
                    url,
                    status_code,
                    sleep_seconds,
                    attempt + 1,
                    retries + 1,
                )
                time.sleep(sleep_seconds)
                attempt += 1
                continue

            log.warning("Request failed for %s: %s", url, str(exc))
            return None
        except Exception as exc:  # pragma: no cover - safety net for unexpected errors
            log.warning("Unexpected error requesting %s: %s", url, str(exc))
            return None

    return None

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
    page_count = 0

    while True:
        # Get the page data using safe request
        if next_url:
            # Ensure pagination URL includes authentication
            authenticated_url = _add_polygon_auth(next_url, api_key)
            log.info("Making paginated request (page %d), has_auth=%s", page_count + 1, "apiKey=" in authenticated_url)
            data = _safe_get_json(authenticated_url)
        else:
            log.info("Making initial request, has_auth=%s", "apiKey" in params)
            data = _safe_get_json(base_url, params=params)

        # Handle page-level failures gracefully
        if data is None:
            if page_count == 0:
                # If we can't get the first page, return empty list
                log.error("Failed to fetch initial page, returning empty universe")
                return []
            else:
                # If we've collected some data but a later page fails, log and break
                log.warning("Failed to fetch page %d, using %d symbols collected so far", page_count + 1, len(tickers))
                break

        page_count += 1
        symbols_in_page = len(data.get("results", []))
        log.info("Processing page %d with %d symbols", page_count, symbols_in_page)

        for result in data.get("results", []):
            symbol = result.get("ticker")
            name = result.get("name")

            # Skip if symbol is invalid (None, empty, or contains special chars)
            if not symbol or not isinstance(symbol, str) or not symbol.strip():
                log.warning("Skipping invalid symbol in results: %s", result)
                continue

            # Try to get market_cap from the paginated result first
            market_cap = result.get("market_cap")

            # Only call the details endpoint if market_cap is missing
            if market_cap is None:
                # URL-encode the symbol for use in path
                encoded_symbol = urllib.parse.quote(symbol.strip(), safe='')
                detail_url = f"https://api.polygon.io/v3/reference/tickers/{encoded_symbol}"
                details = _safe_get_json(detail_url, params={"apiKey": api_key})
                if details is not None:
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

    log.info("Completed universe fetch: %d symbols from %d pages", len(tickers), page_count)
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

        # Make a minimal request to test connectivity
        base_url = "https://api.polygon.io/v3/reference/tickers"
        params = {
            "market": "stocks",
            "active": "true",
            "limit": 1,
            "apiKey": api_key,
        }

        data = _safe_get_json(base_url, params=params)
        if data is not None:
            results_count = len(data.get("results", []))
            log.info("Polygon API test successful, received %d results", results_count)
            return True
        else:
            log.error("Polygon API test failed: no data received")
            return False

    except RuntimeError as e:
        log.warning("Polygon API test skipped: %s", e)
        return False

def _poly_ticker_info() -> Dict[str, Any]:
    """Fetch ticker info from Polygon (placeholder kept for compatibility)."""
    return {}

async def _poly_adv(symbol: str, start: date, end: date) -> float | None:
    """Fetch average daily volume from Polygon (placeholder kept for compatibility)."""
    return None

def rebuild_universe() -> List[Dict[str, Any]]:
    """
    Rebuild the universe table by inserting/updating all symbols with market
    capitalisation below $3B. For each symbol, the function upserts the record
    with included=True in the Universe table.

    Returns
    -------
    List[Dict[str, Any]]
        A list of symbols (dicts with 'symbol' and 'name' keys) that were
        successfully added to the universe. Returns an empty list if no
        symbols are found.

    Raises
    ------
    Exception
        Re-raises any database or connectivity errors for proper error handling
        by callers.
    """
    log.info("Starting universe rebuild: fetching small cap symbols.")
    symbols = _list_small_cap_symbols()
    if not symbols:
        log.warning("No symbols retrieved for universe rebuild.")
        return []

    try:
        # Convert symbols to DataFrame for batch processing
        import pandas as pd
        import db
        import unicodedata

        if SessionLocal is not None:
            session = SessionLocal()
            session.close()

        df_data: List[Dict[str, Any]] = []
        truncated_names = 0

        for item in symbols:
            # Sanitize the company name
            name = item.get("name")
            if name and isinstance(name, str):
                # Normalize Unicode (NFKC) and collapse whitespace
                normalized = unicodedata.normalize('NFKC', name)
                collapsed = ' '.join(normalized.split())

                # Truncate if longer than 256 characters (with a warning)
                if len(collapsed) > 256:
                    collapsed = collapsed[:256]
                    truncated_names += 1

                name = collapsed

            # Build row data; use timezone-aware UTC for last_updated
            df_data.append({
                "symbol": item["symbol"],
                "name": name,
                "included": True,
                # datetime.utcnow() is deprecated and timezone-naive
                "last_updated": datetime.now(timezone.utc),
            })

        if truncated_names > 0:
            log.warning(
                "Truncated %d company names longer than 256 characters during universe rebuild.",
                truncated_names,
            )

        df = pd.DataFrame(df_data)

        # Use upsert_dataframe which handles parameter limits automatically.
        # Do not pass chunk_size; our patched upsert handles chunking internally.
        db.upsert_dataframe(df, Universe, conflict_cols=["symbol"])  # chunking handled in db.upsert_dataframe

        log.info("Universe rebuild completed successfully with %d symbols.", len(symbols))
        return symbols
    except Exception as e:
        log.error("Failed to rebuild universe: %s", e, exc_info=True)
        raise

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

        # Log the results and exit successfully if no exception was raised
        log.info("Universe rebuild completed successfully with %d symbols", len(result))
        sys.exit(0)

    except Exception as e:
        log.error(f"Universe rebuild failed with exception: {e}", exc_info=True)
        sys.exit(1)
