from __future__ import annotations
import logging
import asyncio
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

        # Handle empty or invalid response
        if not js:
            return pd.DataFrame()

        # Check if there's a 'results' field with list data (expected Polygon format)
        if 'results' in js and isinstance(js['results'], list) and len(js['results']) > 0:
            return pd.DataFrame(js['results'])

        # If no results or empty results, return empty DataFrame
        # This handles cases where Polygon returns scalar values like {"status": "OK", "count": 0}
        return pd.DataFrame()

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


def ingest_bars_for_universe(days: int = 30) -> None:
    """
    Ingest daily bars for all symbols in the universe.

    Args:
        days: Number of days of historical data to fetch (default: 30)
    """
    import asyncio
    from datetime import datetime, timedelta

    # Use the existing log variable from module level
    log.info(f"Starting market data ingestion for {days} days")

    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    # Get universe of symbols
    try:
        from data.universe import rebuild_universe
        from db import upsert_dataframe, DailyBar, engine
        from sqlalchemy import text

        # Load universe
        with engine.connect() as conn:
            result = conn.execute(text("SELECT symbol FROM universe WHERE included = true"))
            universe_symbols = [row[0] for row in result.fetchall()]

        if not universe_symbols:
            log.warning("No symbols found in universe, rebuilding...")
            universe_df = rebuild_universe()
            universe_symbols = universe_df['symbol'].tolist()

        if not universe_symbols:
            log.error("No symbols available for ingestion")
            return

        log.info(f"Fetching bars for {len(universe_symbols)} symbols from {start_date} to {end_date}")

        # Try Alpaca first (batch processing)
        alpaca_df = _bars_from_alpaca_batch(universe_symbols, start_date, end_date)

        all_data = []
        if not alpaca_df.empty:
            log.info(f"Retrieved {len(alpaca_df)} bars from Alpaca")
            all_data.append(_normalize_bar_data(alpaca_df, source='alpaca'))

        # For symbols not covered by Alpaca or as fallback, use Polygon
        covered_symbols = set(alpaca_df['symbol'].unique()) if not alpaca_df.empty else set()
        missing_symbols = [s for s in universe_symbols if s not in covered_symbols]

        if missing_symbols and POLYGON_API_KEY:
            log.info(f"Fetching {len(missing_symbols)} symbols from Polygon")
            polygon_df = asyncio.run(_fetch_polygon_daily(missing_symbols, start_date, end_date))
            if not polygon_df.empty:
                log.info(f"Retrieved {len(polygon_df)} bars from Polygon")
                all_data.append(_normalize_bar_data(polygon_df, source='polygon'))

        # Combine and upsert data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            if not combined_df.empty:
                # Remove duplicates (symbol, date)
                combined_df = combined_df.drop_duplicates(subset=['symbol', 'ts'])
                log.info(f"Upserting {len(combined_df)} total bars to database")
                upsert_dataframe(combined_df, DailyBar, ['symbol', 'ts'])
                log.info("Market data ingestion completed successfully")
            else:
                log.warning("No data to upsert")
        else:
            log.warning("No market data retrieved from any source")

    except Exception as e:
        log.error(f"Market data ingestion failed: {e}", exc_info=True)
        raise


def _normalize_bar_data(df: pd.DataFrame, source: str = 'unknown') -> pd.DataFrame:
    """
    Normalize bar data to match DailyBar schema.

    Args:
        df: Raw bar data from provider
        source: Data source ('alpaca', 'polygon', etc.)

    Returns:
        Normalized DataFrame with DailyBar columns
    """
    if df.empty:
        return pd.DataFrame()

    try:
        normalized = df.copy()

        # Standardize column names based on source
        if source == 'alpaca':
            # Alpaca format: symbol, timestamp, open, high, low, close, volume, trade_count, vwap
            if 'timestamp' in normalized.columns:
                normalized['ts'] = pd.to_datetime(normalized['timestamp']).dt.date
            elif 'time' in normalized.columns:
                normalized['ts'] = pd.to_datetime(normalized['time']).dt.date
        elif source == 'polygon':
            # Polygon format: symbol, t (timestamp), o, h, l, c, v, n (trade_count), vw (vwap)
            column_mapping = {
                't': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'n': 'trade_count',
                'vw': 'vwap'
            }
            normalized = normalized.rename(columns=column_mapping)
            if 'timestamp' in normalized.columns:
                # Polygon timestamps are in milliseconds
                normalized['ts'] = pd.to_datetime(normalized['timestamp'], unit='ms').dt.date

        # Ensure required columns exist
        required_columns = ['symbol', 'ts', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in normalized.columns:
                if col == 'volume':
                    normalized[col] = 0
                else:
                    log.warning(f"Missing required column {col} in {source} data")
                    return pd.DataFrame()

        # Add optional columns with defaults
        if 'adj_close' not in normalized.columns:
            normalized['adj_close'] = normalized['close']  # Default to close if no adjustment
        if 'vwap' not in normalized.columns:
            normalized['vwap'] = None
        if 'trade_count' not in normalized.columns:
            normalized['trade_count'] = None

        # Select and order columns to match DailyBar schema
        final_columns = ['symbol', 'ts', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'vwap', 'trade_count']
        normalized = normalized[final_columns]

        # Ensure proper data types
        numeric_columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume', 'vwap', 'trade_count']
        for col in numeric_columns:
            if col in normalized.columns:
                normalized[col] = pd.to_numeric(normalized[col], errors='coerce')

        # Remove any rows with missing critical data
        normalized = normalized.dropna(subset=['symbol', 'ts', 'open', 'high', 'low', 'close'])

        return normalized

    except Exception as e:
        log.error(f"Error normalizing {source} data: {e}", exc_info=True)
        return pd.DataFrame()
