
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import List, Optional
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
from db import engine, SessionLocal, DailyBar, upsert_dataframe
from db import Universe  # type: ignore
from market_calendar import get_previous_market_day, is_market_day

log = logging.getLogger("data.ingest")


def _resolve_target_end_date(explicit_end: Optional[date] = None) -> date:
    """
    Decide the last data date we should collect for, accounting for market hours.
    - If PIPELINE_TARGET_DATE is set: use that (string YYYY-MM-DD).
    - Else: if today is before close, use previous market day; otherwise today.
    """
    import os
    from datetime import datetime

    env_td = os.getenv("PIPELINE_TARGET_DATE")
    if env_td:
        try:
            return datetime.strptime(env_td, "%Y-%m-%d").date()
        except Exception:
            log.warning("Invalid PIPELINE_TARGET_DATE=%s; ignoring", env_td)

    if explicit_end:
        return explicit_end

    today = date.today()
    now = datetime.now()
    if is_market_day(today) and now.hour < 16:
        return get_previous_market_day(today)
    return today


def _universe_symbols(limit: Optional[int] = None) -> List[str]:
    """
    Get the trading universe (included = True). If empty, return a small fallback.
    """
    try:
        with SessionLocal() as session:
            q = session.query(Universe.symbol)
            if hasattr(Universe, "included"):
                q = q.filter(Universe.included.is_(True))
            if limit and limit > 0:
                q = q.limit(limit)
                        rows = q.all()
        
                        syms = [r[0] for r in rows]
                 if syms:
                return syms
    except Exception as e:
        log.warning("Failed to load universe from DB: %s", e)

    fallback = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"]
    log.warning("Universe is empty; using fallback symbols: %s", fallback)
    return fallback


def _standardize_bar_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has columns: symbol, ts, open, high, low, close, adj_close, volume, vwap, trade_count.
    Extra columns are dropped; missing become NaN/None; ts is coerced to date.
    """
    cols_out = ["symbol", "ts", "open", "high", "low", "close", "adj_close", "volume", "vwap", "trade_count"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols_out)

    df = df.copy()

    # Normalize ts
    if "ts" not in df.columns:
        for cand in ("timestamp", "time", "t", "datetime"):
            if cand in df.columns:
                try:
                    ts_series = pd.to_datetime(df[cand], utc=True, errors="coerce")
                except Exception:
                    ts_series = pd.to_datetime(df[cand], errors="coerce")
                df["ts"] = ts_series.dt.date
                break

    if "ts" not in df.columns:
        log.warning("Standardization could not find/convert timestamp; dropping DataFrame")
        return pd.DataFrame(columns=cols_out)

    # Ensure symbol exists
    if "symbol" not in df.columns:
        if isinstance(df.index, pd.MultiIndex) and "symbol" in df.index.names:
            df = df.reset_index("symbol")
        else:
            log.warning("Standardization missing symbol column; dropping DataFrame")
            return pd.DataFrame(columns=cols_out)

    # Map provider keys
    rename_map = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "tradecount": "trade_count",
        "trade_count": "trade_count",
        "n": "trade_count",  # Polygon uses "n" for trade count
        "vw": "vwap",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    if "adj_close" not in df.columns:
        df["adj_close"] = df.get("close")

    keep_cols = [c for c in cols_out if c in df.columns]
    df = df[keep_cols].copy()

    # Required fields presence
    for c in ("symbol", "ts", "close"):
        if c not in df.columns:
            df[c] = None
    df = df.dropna(subset=["symbol", "ts", "close"])

    df["symbol"] = df["symbol"].astype(str)
    try:
        df["ts"] = pd.to_datetime(df["ts"]).dt.date
    except Exception:
        pass

    # Dedup
    before = len(df)
    df = df.drop_duplicates(subset=["symbol", "ts"], keep="last").reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        log.debug("Removed %d duplicate bar rows during standardization", removed)

    # Ensure full column set
    for c in cols_out:
        if c not in df.columns:
            df[c] = None

    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    return df[cols_out]


def _fetch_from_alpaca(symbols: List[str], start: date, end: date) -> pd.DataFrame:
    """
    Fetch daily bars from Alpaca across the symbol list. Returns standardized DataFrame.
    """
    if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY or not APCA_API_BASE_URL:
        return pd.DataFrame()

    try:
        import alpaca_trade_api as tradeapi
        from alpaca_trade_api.rest import TimeFrame

        api = tradeapi.REST(
            key_id=APCA_API_KEY_ID,
            secret_key=APCA_API_SECRET_KEY,
            base_url=APCA_API_BASE_URL,
        )

        chunk = 200
        dfs: List[pd.DataFrame] = []
        for i in range(0, len(symbols), chunk):
            batch = symbols[i:i + chunk]
            try:
                raw = api.get_bars(
                    batch,
                    TimeFrame.Day,
                    start.isoformat(),
                    end.isoformat(),
                    adjustment="all",
                    feed=ALPACA_DATA_FEED or "iex",
                ).df
            except Exception as e:
                log.error("Alpaca get_bars failed for batch %d-%d: %s", i, i + len(batch), e)
                continue

            if raw is None or raw.empty:
                continue

            raw = raw.reset_index()
            if "timestamp" in raw.columns:
                raw["ts"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce").dt.date
            elif "time" in raw.columns:
                raw["ts"] = pd.to_datetime(raw["time"], utc=True, errors="coerce").dt.date

            raw_std = _standardize_bar_df(raw)
            if not raw_std.empty:
                dfs.append(raw_std)

        if dfs:
            out = pd.concat(dfs, ignore_index=True)
            log.info("Alpaca fetched %d rows across %d symbols", len(out), out["symbol"].nunique())
            return out
        return pd.DataFrame()

    except ImportError as e:
        log.warning("alpaca_trade_api not installed; skipping Alpaca provider: %s", e)
        return pd.DataFrame()
    except Exception as e:
        log.error("Unexpected Alpaca error: %s", e, exc_info=True)
        return pd.DataFrame()


def _fetch_from_polygon(symbols: List[str], start: date, end: date) -> pd.DataFrame:
    """
    Fetch daily bars from Polygon across the symbol list (synchronous).
    """
    if not POLYGON_API_KEY:
        log.warning("POLYGON_API_KEY not set; skipping Polygon data fetch")
        return pd.DataFrame()

    log.info("Starting Polygon data fetch for %d symbols, has_api_key=%s", len(symbols), bool(POLYGON_API_KEY))

    import time
    import requests

    dfs: List[pd.DataFrame] = []
    for idx, sym in enumerate(symbols):
        url = f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/{start.isoformat()}/{end.isoformat()}"
        params = {"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY}
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(1.5)
                r = requests.get(url, params=params, timeout=15)

            if r.status_code != 200:
                if r.status_code == 401:
                    log.error(
                        "Polygon returned 401 Unauthorized for %s. Check POLYGON_API_KEY in the environment "
                        "for this runtime (e.g., Render) and ensure auth is included on requests.", sym
                    )
                else:
                    log.warning("Polygon %s returned %s", sym, r.status_code)
                continue

            data = r.json()
            if not data or "results" not in data:
                continue

            rows = []
            for result in data.get("results", []):
                rows.append({
                    "symbol": sym.upper(),
                    "ts": pd.to_datetime(result.get("t"), unit="ms").date(),
                    "open": result.get("o"),
                    "high": result.get("h"),
                    "low": result.get("l"),
                    "close": result.get("c"),
                    "volume": result.get("v"),
                    "adj_close": result.get("c"),  # adjusted=true returns adjusted close
                    "vwap": result.get("vw"),
                    "trade_count": result.get("n"),
                })

            if rows:
                df_sym = pd.DataFrame(rows)
                df_std = _standardize_bar_df(df_sym)
                if not df_std.empty:
                    dfs.append(df_std)

            # Rate limiting
            if idx > 0 and idx % 10 == 0:
                time.sleep(0.5)

        except Exception as e:
            log.error("Error fetching Polygon data for %s: %s", sym, e)
            continue

    if dfs:
        out = pd.concat(dfs, ignore_index=True)
        log.info("Polygon fetched %d rows across %d symbols", len(out), out["symbol"].nunique())
        return out
    return pd.DataFrame()


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
        log.debug("POLYGON_API_KEY not set; skipping Polygon data fetch for %s", symbol)
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


_ADJ_CLOSE_CHECKED = False

def _ensure_adj_close_column() -> None:
    """Ensure the daily_bars table has an adj_close column.

    Older databases might have been created before the adj_close field
    existed.  This helper adds the column on-the-fly if it's missing so
    downstream feature building can rely on it.  The check is cached so
    we only touch the database once per process.
    """
    global _ADJ_CLOSE_CHECKED
    if _ADJ_CLOSE_CHECKED:
        return
    from sqlalchemy import inspect, text
    try:
        insp = inspect(engine)
        cols = {c['name'] for c in insp.get_columns('daily_bars')}
        if 'adj_close' not in cols:
            with engine.begin() as conn:
                conn.execute(text('ALTER TABLE daily_bars ADD COLUMN adj_close DOUBLE PRECISION'))
            log.warning("Added missing adj_close column to daily_bars")
    except Exception:
        # If inspection fails we silently continue; subsequent logic will
        # fall back to using close prices.
        pass
    finally:
        _ADJ_CLOSE_CHECKED = True

def _upsert_daily_bars(df: pd.DataFrame, chunk_size: int = 5000) -> int:
    """Upsert to daily_bars using PostgreSQL ON CONFLICT."""
    if df is None or df.empty:
        return 0
    _ensure_adj_close_column()
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
    """
    Fetch and upsert daily bars for investable universe from Alpaca (pref) or Polygon (fallback).
    Uses market calendar for safe date resolution and db.upsert_dataframe for robust database operations.
    """
    if days <= 0:
        raise ValueError("days must be positive")

    # Resolve safe end date using market calendar and PIPELINE_TARGET_DATE
    end = _resolve_target_end_date()
    start = end - timedelta(days=days)

    # Load universe symbols with fallback
    symbols = _universe_symbols()
    if not symbols:
        # If universe loading failed, use the legacy fallback
        try:
            symbols = _fallback_symbols_from_alpaca(max_symbols=200)
        except Exception as e:
            log.warning("Alpaca fallback failed: %s", e)

    if not symbols:
        raise RuntimeError("No symbols available from universe or Alpaca assets. Populate universe first.")

    log.info("Ingesting daily bars for %d symbols from %s to %s", len(symbols), start, end)

    all_dfs = []

    # Prefer Alpaca if configured
    if APCA_API_KEY_ID:
        log.info("Using Alpaca as primary data source")
        alpaca_df = _fetch_from_alpaca(symbols, start, end)
        if not alpaca_df.empty:
            all_dfs.append(alpaca_df)
        else:
            log.warning("Alpaca returned no data, will try Polygon as fallback")

    # Fall back to Polygon if Alpaca failed or is not configured
    if not all_dfs and POLYGON_API_KEY:
        log.info("Using Polygon as data source")
        polygon_df = _fetch_from_polygon(symbols, start, end)
        if not polygon_df.empty:
            all_dfs.append(polygon_df)
        else:
            log.warning("Polygon also returned no data")

    if not all_dfs:
        log.warning("No data retrieved from any provider")
        return False

    # Combine all data and standardize
    combined_df = pd.concat(all_dfs, ignore_index=True)
    if combined_df.empty:
        log.warning("Combined data is empty")
        return False

    # Final standardization and deduplication
    final_df = _standardize_bar_df(combined_df)

    if final_df.empty:
        log.warning("No valid data after standardization")
        return False

    # Use the robust upsert_dataframe function instead of custom PostgreSQL-only code
    try:
        log.info("Upserting %d rows to daily_bars", len(final_df))
        upsert_dataframe(final_df, DailyBar, conflict_cols=["symbol", "ts"])
        log.info("Ingestion complete. Upserted %d rows", len(final_df))
        return True
    except Exception as e:
        log.error("Failed to upsert data: %s", e, exc_info=True)
        return False


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
