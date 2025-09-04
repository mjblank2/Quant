from __future__ import annotations
import logging, asyncio
from datetime import date
from typing import List
import pandas as pd

from config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, ALPACA_DATA_FEED, POLYGON_API_KEY
from utils_http import get_json_async

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
        return df  # Processing omitted here for brevity
    except Exception as e:
        log.error(f"Error fetching Alpaca bars: {e}", exc_info=True)
        return pd.DataFrame()

async def _fetch_polygon_daily_one(symbol: str, start: date, end: date) -> pd.DataFrame:
    if not POLYGON_API_KEY:
        return pd.DataFrame()
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    try:
        js = await get_json_async(url, params={"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY})
        return pd.DataFrame(js or {})
    except Exception as e:
        log.error(f"Error fetching/processing Polygon data for {symbol}: {e}", exc_info=True)
        return pd.DataFrame()

async def _fetch_polygon_daily(symbols: List[str], start: date, end: date) -> pd.DataFrame:
    batch_size = 100
    all_dfs = []
    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i+batch_size]
        tasks = [_fetch_polygon_daily_one(s, start, end) for s in batch_symbols]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in batch_results:
            if isinstance(result, Exception):
                log.error(f"A task in the Polygon batch failed: {result}")
            elif result is not None and not result.empty:
                all_dfs.append(result)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
