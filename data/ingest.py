from __future__ import annotations
from datetime import date, timedelta
from typing import List
import asyncio
import pandas as pd
from sqlalchemy import text
from db import engine, upsert_dataframe, DailyBar
from config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, ALPACA_DATA_FEED, TIINGO_API_KEY, POLYGON_API_KEY
from utils_http import get_json, get_json_async


def _bars_from_alpaca_batch(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    try:
        import alpaca_trade_api as tradeapi
        from alpaca_trade_api.rest import TimeFrame
        api = tradeapi.REST(key_id=APCA_API_KEY_ID, secret_key=APCA_API_SECRET_KEY, base_url=APCA_API_BASE_URL)
        df = api.get_bars(symbols, TimeFrame.Day, start.isoformat(), end.isoformat(), adjustment='all', feed=ALPACA_DATA_FEED).df
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(columns={"timestamp": "ts"})
        df["ts"] = pd.to_datetime(df["ts"]).dt.date
        df["adj_close"] = df["close"]
        use = ["symbol", "ts", "open", "high", "low", "close", "adj_close", "volume", "vwap", "trade_count"]
        return df[use]
    except Exception:
        return pd.DataFrame()


async def _fetch_polygon_daily_one(symbol: str, start: date, end: date) -> pd.DataFrame:
    if not POLYGON_API_KEY:
        return pd.DataFrame()
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    js = await get_json_async(url, params={"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY})
    try:
        res = js.get("results") or []
        if not res:
            return pd.DataFrame()
        df = pd.DataFrame([{
            "symbol": symbol,
            "ts": pd.to_datetime(r["t"], unit="ms").date(),
            "open": r.get("o"),
            "high": r.get("h"),
            "low":  r.get("l"),
            "close": r.get("c"),
            "adj_close": r.get("c"),
            "volume": r.get("v"),
            "vwap": r.get("vw"),
            "trade_count": r.get("n"),
        } for r in res])
        return df
    except Exception:
        return pd.DataFrame()


async def _fetch_polygon_daily(symbols: List[str], start: date, end: date) -> pd.DataFrame:
    # Process symbols in smaller batches to reduce memory pressure
    batch_size = 100  # Reduce concurrent requests to limit memory usage
    all_dfs = []

    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i+batch_size]
        tasks = [_fetch_polygon_daily_one(s, start, end) for s in batch_symbols]
        batch_results = await asyncio.gather(*tasks)

        # Process batch results immediately to reduce memory accumulation
        batch_dfs = [d for d in batch_results if d is not None and not d.empty]
        if batch_dfs:
            batch_combined = pd.concat(batch_dfs, ignore_index=True)
            all_dfs.append(batch_combined)
            # Clear batch data to free memory
            del batch_dfs, batch_combined

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def _fetch_tiingo_daily(symbol: str, start: date, end: date) -> pd.DataFrame:
    if not TIINGO_API_KEY:
        return pd.DataFrame()
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    js = get_json(url, params={'startDate': start.isoformat(), 'endDate': end.isoformat(), 'token': TIINGO_API_KEY})
    try:
        if not js:
            return pd.DataFrame()
        df = pd.DataFrame(js)
        if 'date' not in df.columns:
            return pd.DataFrame()
        df['ts'] = pd.to_datetime(df['date']).dt.date
        df['adj_close'] = df['adjClose'] if 'adjClose' in df.columns else df['close']
        keep = ['ts', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        df = df[[c for c in keep if c in df.columns]].copy()
        df['symbol'] = symbol
        return df
    except Exception:
        return pd.DataFrame()


def ingest_bars_for_universe(days: int = 365) -> None:
    import gc  # Import garbage collector for explicit memory management

    with engine.connect() as con:
        uni = pd.read_sql_query(text("SELECT symbol FROM universe WHERE included = TRUE"), con)
    symbols = uni['symbol'].tolist()
    if not symbols:
        return

    end = pd.Timestamp('today').normalize().date()
    start = end - timedelta(days=int(days*1.2))

    # Use smaller batches for memory efficiency during intensive operations
    alpaca_batch_size = 200 if len(symbols) > 1000 else 300  # Reduce batch size for large universes

    # 1) Alpaca batch in chunks
    fetched = []
    for i in range(0, len(symbols), alpaca_batch_size):
        subs = symbols[i:i+alpaca_batch_size]
        df = _bars_from_alpaca_batch(subs, start, end)
        if not df.empty:
            upsert_dataframe(df, DailyBar, ['symbol', 'ts'])
            fetched.extend(df['symbol'].unique().tolist())
        # Explicit garbage collection after each batch to free memory
        gc.collect()

    remaining = sorted(set(symbols) - set(fetched))

    # 2) Polygon async (batch per symbol)
    if remaining and POLYGON_API_KEY:
        poly_df = asyncio.run(_fetch_polygon_daily(remaining, start, end))
        if not poly_df.empty:
            upsert_dataframe(poly_df, DailyBar, ['symbol', 'ts'])
            remain2 = sorted(set(remaining) - set(poly_df['symbol'].unique().tolist()))
            # Clear large DataFrame after use
            del poly_df
            gc.collect()
        else:
            remain2 = remaining
    else:
        remain2 = remaining

    # 3) Tiingo per symbol - process in smaller batches to reduce memory usage
    tiingo_batch_size = 50  # Process 50 symbols at a time to limit memory usage
    for batch_start in range(0, len(remain2), tiingo_batch_size):
        batch_symbols = remain2[batch_start:batch_start + tiingo_batch_size]
        batch_rows = []

        for sym in batch_symbols:
            df = _fetch_tiingo_daily(sym, start, end)
            if not df.empty:
                batch_rows.append(df)

        if batch_rows:
            batch_df = pd.concat(batch_rows, ignore_index=True).drop_duplicates(subset=['symbol', 'ts'])
            batch_df['vwap'] = None
            batch_df['trade_count'] = None
            upsert_dataframe(batch_df, DailyBar, ['symbol', 'ts'])
            # Clear batch data to free memory
            del batch_rows, batch_df
            gc.collect()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=365)
    args = p.parse_args()
    ingest_bars_for_universe(args.days)
