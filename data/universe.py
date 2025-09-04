# =============================================================================
# Module: data/universe.py
# =============================================================================
import logging
from datetime import date
from typing import Dict, Any
import pandas as pd

try:
    from config import POLYGON_API_KEY
except Exception:
    POLYGON_API_KEY = None

try:
    from utils_http import get_json_async
except Exception:
    async def get_json_async(*args, **kwargs):
        return None

log_universe = logging.getLogger("data.universe")

def _list_alpaca_assets() -> pd.DataFrame:
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST()
        assets = api.list_assets()
        rows = []
        for a in assets:
            rows.append({"symbol": a.symbol, "name": getattr(a, "name", None), "exchange": a.exchange})
        return pd.DataFrame(rows).drop_duplicates(subset=['symbol'])
    except Exception as e:
        log_universe.error(f"Failed to list Alpaca assets: {e}", exc_info=True)
        return pd.DataFrame(columns=['symbol','name','exchange'])

async def _poly_ticker_info(symbol: str) -> Dict[str, Any]:
    if not POLYGON_API_KEY:
        return {}
    try:
        url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
        return await get_json_async(url, params={"apiKey": POLYGON_API_KEY}) or {}
    except Exception as e:
        log_universe.error(f"Error fetching Polygon ticker info for {symbol}: {e}", exc_info=True)
        return {}

async def _poly_adv(symbol: str, start: date, end: date) -> float | None:
    if not POLYGON_API_KEY:
        return None
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}"
        js = await get_json_async(url, params={"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY})
        if not js or "results" not in js or not js["results"]:
            return None
        df = pd.DataFrame(js["results"])
        dv = df.get("v", pd.Series(dtype=float)).mean()
        return float(dv) if pd.notnull(dv) else None
    except Exception as e:
        log_universe.error(f"Error calculating Polygon ADV for {symbol}: {e}", exc_info=True)
        return None

def rebuild_universe() -> pd.DataFrame:
    """
    Build the investable universe and persist it if DB utilities are available.
    Returns a DataFrame with at least: symbol, name, exchange, included.
    """
    log = log_universe
    df = _list_alpaca_assets()
    if df.empty:
        log.warning("No assets retrieved from Alpaca; returning empty universe.")
        return df

    # Normalize and default flags
    try:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    except Exception:
        pass
    if "included" not in df.columns:
        df["included"] = True  # default include; downstream filters can refine this

    # Attempt to persist to DB if helpers/models exist
    try:
        from db import upsert_dataframe, Universe  # type: ignore
        cols = [c for c in ["symbol", "name", "exchange", "included"] if c in df.columns]
        upsert_dataframe(df[cols], Universe, ["symbol"])
        log.info("Universe upserted to DB with %d symbols.", len(df))
    except Exception as e:
        # Non-fatal: still return the DataFrame so callers can proceed
        log.warning(f"DB upsert skipped or failed: {e}")

    return df
