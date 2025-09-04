
from __future__ import annotations

import logging
from datetime import date
from typing import List, Dict, Any
import pandas as pd

try:
    from config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, ALPACA_DATA_FEED, POLYGON_API_KEY
except Exception:
    APCA_API_KEY_ID = APCA_API_SECRET_KEY = APCA_API_BASE_URL = ALPACA_DATA_FEED = POLYGON_API_KEY = None

try:
    from utils_http import get_json, get_json_async  # type: ignore
except Exception:
    def get_json(*a, **kw): return None
    async def get_json_async(*a, **kw): return None

log = logging.getLogger("data.ingest")

def _bars_from_alpaca_batch(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    if not APCA_API_KEY_ID:
        return pd.DataFrame()
    try:
        import alpaca_trade_api as tradeapi
        from alpaca_trade_api.rest import TimeFrame
        api = tradeapi.REST(key_id=APCA_API_KEY_ID, secret_key=APCA_API_SECRET_KEY, base_url=APCA_API_BASE_URL)
        df = api.get_bars(symbols, TimeFrame.Day, start.isoformat(), end.isoformat(), adjustment='all', feed=ALPACA_DATA_FEED).df
        if df is None or df.empty:
            return pd.DataFrame()
        # caller takes care of further processing
        return df.reset_index()
    except Exception as e:
        log.error(f"Error fetching Alpaca bars: {e}", exc_info=True)
        return pd.DataFrame()

async def _fetch_polygon_daily_one(symbol: str, start: date, end: date) -> pd.DataFrame:
    if not POLYGON_API_KEY:
        return pd.DataFrame()
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    try:
        js = await get_json_async(url, params={"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY})
        if not js or 'results' not in js:
            return pd.DataFrame()
        rows = []
        for r in js.get('results', []):
            rows.append({
                "symbol": symbol,
                "ts": pd.to_datetime(r.get('t'), unit='ms').date(),
                "open": r.get('o'),
                "high": r.get('h'),
                "low": r.get('l'),
                "close": r.get('c'),
                "volume": r.get('v'),
                "adj_close": r.get('c'),  # Polygon returns adjusted by default with adjusted=true
            })
        return pd.DataFrame(rows)
    except Exception as e:
        log.error(f"Error fetching/processing Polygon data for {symbol}: {e}", exc_info=True)
        return pd.DataFrame()

async def _fetch_polygon_daily(symbols: List[str], start: date, end: date) -> pd.DataFrame:
    import asyncio
    batch_size = 100
    all_dfs: List[pd.DataFrame] = []
    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i+batch_size]
        tasks = [_fetch_polygon_daily_one(s, start, end) for s in batch_symbols]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        batch_dfs = []
        for result in batch_results:
            if isinstance(result, Exception):
                log.error(f"A task in the Polygon batch failed: {result}")
            elif result is not None and isinstance(result, pd.DataFrame) and not result.empty:
                batch_dfs.append(result)
        if batch_dfs:
            all_dfs.append(pd.concat(batch_dfs, ignore_index=True))
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def ingest_bars_for_universe(days: int = 7) -> bool:
    """Placeholder to orchestrate data ingestion (customize to your data providers)."""
    log.info("Ingestion run placeholder. Implement provider-specific logic here.")
    return True
