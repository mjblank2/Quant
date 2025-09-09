
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import List, Dict, Any, Iterable, Optional
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

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import select
from db import engine, SessionLocal, DailyBar
from db import Universe  # type: ignore

log = logging.getLogger("data.ingest")


def _get_universe_symbols() -> List[str]:
    """Load investable symbols from DB universe (included = true)."""
    try:
        with SessionLocal() as s:
            rows = s.execute(select(Universe.symbol).where(Universe.included == True)).scalars().all()  # noqa: E712
        syms = sorted(set([r.strip().upper() for r in rows if r]))
        if syms:
            log.info(f"Loaded {len(syms)} symbols from universe.")
        return syms
    except Exception as e:
        log.error(f"Failed to load universe symbols: {e}", exc_info=True)
        return []


def _fallback_symbols_from_alpaca(max_symbols: int = 200) -> List[str]:
    """If universe is empty, use Alpaca assets list as a fallback (real, not placeholder)."""
    if not APCA_API_KEY_ID:
        return []
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(key_id=APCA_API_KEY_ID, secret_key=APCA_API_SECRET_KEY, base_url=APCA_API_BASE_URL)
        assets = api.list_assets(status="active")
        # Filter tradable US equities
        syms = [a.symbol for a in assets if getattr(a, "tradable", False) and getattr(a, "asset_class", "us_equity") == "us_equity"]
        syms = sorted(set(syms))[:max_symbols]
        if syms:
            log.warning(f"Universe empty; falling back to {len(syms)} Alpaca assets.")
        return syms
    except Exception as e:
        log.error(f"Failed to load fallback symbols from Alpaca: {e}", exc_info=True)
        return []


def _normalize_alpaca_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Alpaca bars to DailyBar schema."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "adj_close", "volume", "vwap", "trade_count"])
    # After reset_index(), columns typically include ['timestamp','symbol','open','high','low','close','volume','vwap','trade_count']
    df = df.rename(columns={"timestamp": "ts"})
    if pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = df["ts"].dt.date
    else:
        df["ts"] = pd.to_datetime(df["ts"]).dt.date
    # We requested adjustment='all' so close should be adjusted; still persist both close and adj_close
    df["adj_close"] = df["close"]
    keep_cols = ["symbol", "ts", "open", "high", "low", "close", "adj_close", "volume", "vwap", "trade_count"]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None
    out = df[keep_cols].copy()
    # ensure types
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype(int)
    for c in ["open", "high", "low", "close", "adj_close", "vwap"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["trade_count"] = pd.to_numeric(out["trade_count"], errors="coerce").astype("Int64")
    return out


def _bars_from_alpaca_batch(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    if not APCA_API_KEY_ID:
        return pd.DataFrame()
    try:
        import alpaca_trade_api as tradeapi
        from alpaca_trade_api.rest import TimeFrame
        api = tradeapi.REST(key_id=APCA_API_KEY_ID, secret_key=APCA_API_SECRET_KEY, base_url=APCA_API_BASE_URL)
        df = api.get_bars(
            symbols,
            TimeFrame.Day,
            start.isoformat(),
            end.isoformat(),
            adjustment="all",
            feed=ALPACA_DATA_FEED
        ).df
        if df is None or df.empty:
            return pd.DataFrame()
        return _normalize_alpaca_df(df.reset_index())
    except Exception as e:
        log.error(f"Error fetching Alpaca bars: {e}", exc_info=True)
        return pd.DataFrame()


async def _fetch_polygon_daily_one(symbol: str, start: date, end: date) -> pd.DataFrame:
    if not POLYGON_API_KEY:
        return pd.DataFrame()
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    try:
        js = await get_json_async(url, params={"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY})
        if not js or "results" not in js:
            return pd.DataFrame()
        rows = []
        for r in js.get("results", []):
            rows.append({
                "symbol": symbol.upper(),
                "ts": pd.to_datetime(r.get("t"), unit="ms").date(),
                "open": r.get("o"),
                "high": r.get("h"),
                "low": r.get("l"),
                "close": r.get("c"),
                "volume": r.get("v"),
                "adj_close": r.get("c"),   # adjusted=true returns adjusted close
                "vwap": r.get("vw"),
                "trade_count": r.get("n"),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # ensure types
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        for c in ["open", "high", "low", "close", "adj_close", "vwap"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce").astype("Int64")
        return df
    except Exception as e:
        log.error(f"Error fetching/processing Polygon data for {symbol}: {e}", exc_info=True)
        return pd.DataFrame()


async def _fetch_polygon_daily(symbols: List[str], start: date, end: date) -> pd.DataFrame:
    import asyncio
    batch_size = 100
    all_dfs: List[pd.DataFrame] = []
    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i + batch_size]
        tasks = [_fetch_polygon_daily_one(s, start, end) for s in batch_symbols]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        batch_dfs: List[pd.DataFrame] = []
        for result in batch_results:
            if isinstance(result, Exception):
                log.error(f"A task in the Polygon batch failed: {result}")
            elif result is not None and isinstance(result, pd.DataFrame) and not result.empty:
                batch_dfs.append(result)
        if batch_dfs:
            all_dfs.append(pd.concat(batch_dfs, ignore_index=True))
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def _dedupe_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate (symbol, ts) to avoid ON CONFLICT double updates."""
    if df is None or df.empty:
        return df
    before = len(df)
    df = df.sort_values(["symbol", "ts"]).drop_duplicates(subset=["symbol", "ts"], keep="last")
    after = len(df)
    if after < before:
        log.info(f"De-duplicated bars: {before} -> {after}")
    return df


def _upsert_daily_bars(df: pd.DataFrame, chunk_size: int = 5000) -> int:
    """Upsert to daily_bars using PostgreSQL ON CONFLICT."""
    if df is None or df.empty:
        return 0
    df = _dedupe_bars(df)
    cols = ["symbol", "ts", "open", "high", "low", "close", "adj_close", "volume", "vwap", "trade_count"]
    df = df[cols].copy()
    total = 0
    table = DailyBar.__table__
    with engine.begin() as conn:
        for start_idx in range(0, len(df), chunk_size):
            chunk = df.iloc[start_idx:start_idx + chunk_size]
            payload = chunk.to_dict(orient="records")
            stmt = pg_insert(table).values(payload)
            update_dict = {
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "adj_close": stmt.excluded.adj_close,
                "volume": stmt.excluded.volume,
                "vwap": stmt.excluded.vwap,
                "trade_count": stmt.excluded.trade_count,
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "ts"],
                set_=update_dict,
            )
            conn.execute(stmt)
            total += len(chunk)
    return total


def ingest_bars_for_universe(days: int = 7) -> bool:
    """Fetch and upsert daily bars for investable universe from Alpaca (pref) or Polygon (fallback)."""
    if days <= 0:
        raise ValueError("days must be positive")

    end = date.today()
    start = end - timedelta(days=days)
    symbols = _get_universe_symbols()
    if not symbols:
        symbols = _fallback_symbols_from_alpaca(max_symbols=200)

    if not symbols:
        raise RuntimeError("No symbols available from universe or Alpaca assets. Populate universe first.")

    log.info(f"Ingesting daily bars for {len(symbols)} symbols from {start} to {end}.")

    all_rows = 0

    # Prefer Alpaca if configured
    if APCA_API_KEY_ID:
        batch = 200
        for i in range(0, len(symbols), batch):
            syms = symbols[i:i + batch]
            df = _bars_from_alpaca_batch(syms, start, end)
            if df is None or df.empty:
                log.warning(f"Alpaca returned no data for batch {i // batch + 1}.")
                continue
            ing = _upsert_daily_bars(df)
            all_rows += ing
            log.info(f"Alpaca batch {i // batch + 1}: upserted {ing} rows.")
    else:
        log.info("Alpaca credentials not found; using Polygon.")
        try:
            import asyncio
            df_poly = asyncio.run(_fetch_polygon_daily(symbols, start, end))
        except RuntimeError:
            # In case we're already in an event loop (e.g., Streamlit), use nested loop approach
            import nest_asyncio, asyncio as aio
            nest_asyncio.apply()
            df_poly = aio.get_event_loop().run_until_complete(_fetch_polygon_daily(symbols, start, end))
        if df_poly is None or df_poly.empty:
            log.warning("Polygon returned no data.")
        else:
            ing = _upsert_daily_bars(df_poly)
            all_rows += ing
            log.info(f"Polygon: upserted {ing} rows.")

    log.info(f"Ingestion complete. Total rows upserted: {all_rows}")
    return True


if __name__ == "__main__":
    import sys
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ingest market data for universe")
    parser.add_argument(
        "--days", 
        type=int, 
        default=7,
        help="Number of days to ingest (default: 7)"
    )
    args = parser.parse_args()
    
    try:
        log.info(f"Starting data ingestion for {args.days} days...")
        result = ingest_bars_for_universe(days=args.days)
        
        if result is True or result is None:
            log.info("Data ingestion completed successfully")
            sys.exit(0)
        else:
            log.error("Data ingestion failed")
            sys.exit(1)
            
    except Exception as e:
        log.error(f"Data ingestion failed with exception: {e}", exc_info=True)
        sys.exit(1)
