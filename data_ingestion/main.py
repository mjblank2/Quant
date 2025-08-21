from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Iterable, List, Dict, Any

import requests
import pandas as pd
from flask import Flask, request, jsonify

# Updated imports for alpaca-py and Enums
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment
from alpaca.common.exceptions import APIError

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from tenacity import retry, stop_after_attempt, wait_exponential

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration and Initialization
# ---------------------------------------------------------------------------

ALPACA_API_KEY: str | None = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY: str | None = os.environ.get("ALPACA_SECRET_KEY")
TIINGO_API_KEY: str | None = os.environ.get("TIINGO_API_KEY")

DATABASE_URL: str | None = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable must be set")

# Determine the Alpaca feed string and map it to the corresponding Enum
ALPACA_FEED_STR: str = os.environ.get("ALPACA_FEED", "sip").strip().lower()
ALPACA_FEED_ENUM: DataFeed = {
    "sip": DataFeed.SIP,
    "iex": DataFeed.IEX,
    "otc": DataFeed.OTC
}.get(ALPACA_FEED_STR, DataFeed.SIP)


# Universe of symbols.
_stock_list: str | None = os.environ.get("STOCK_UNIVERSE")
if _stock_list:
    SYMBOLS: List[str] = [s.strip().upper() for s in _stock_list.split(",") if s.strip()]
else:
    SYMBOLS: List[str] = [
        "AAPL", "MSFT", "SPY", "QQQ", "PLTR", "SMCI", "CRWD", "DDOG",
    ]


# ---------------------------------------------------------------------------
# Database utilities
# ---------------------------------------------------------------------------

def _psycopg3_url(url: str) -> str:
    if url.startswith("postgresql+psycopg://"): return url
    if url.startswith("postgresql://"): return "postgresql+psycopg://" + url[len("postgresql://"):]
    if url.startswith("postgres://"): return "postgresql+psycopg://" + url[len("postgres://"):]
    return url

def _get_engine() -> Engine:
    dsn = _psycopg3_url(DATABASE_URL)
    return create_engine(dsn, pool_pre_ping=True)

def _ensure_schema(engine: Engine) -> None:
    """Create the `daily_bars` table and necessary indexes if they do not exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS daily_bars (
        id BIGSERIAL PRIMARY KEY,
        symbol TEXT NOT NULL,
        ts TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        open DOUBLE PRECISION NOT NULL,
        high DOUBLE PRECISION NOT NULL,
        low DOUBLE PRECISION NOT NULL,
        close DOUBLE PRECISION NOT NULL,
        volume BIGINT NOT NULL,
        trade_count INTEGER,
        vwap DOUBLE PRECISION,
        UNIQUE(symbol, ts)
    );

    -- Add index to optimize read queries (e.g., fetching time series for a symbol)
    CREATE INDEX IF NOT EXISTS ix_daily_bars_symbol_ts_desc ON daily_bars (symbol, ts DESC);
    """
    with engine.begin() as connection:
        connection.exec_driver_sql(ddl)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
def _insert_rows(engine: Engine, rows: Iterable[Dict[str, Any]]) -> int:
    """Insert or update rows using bulk UPSERT."""
    rows_list = list(rows)
    if not rows_list:
        return 0
    sql = text(
        """
        INSERT INTO daily_bars (symbol, ts, open, high, low, close, volume, trade_count, vwap)
        VALUES (:symbol, :ts, :open, :high, :low, :close, :volume, :trade_count, :vwap)
        ON CONFLICT (symbol, ts) DO UPDATE SET
            open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
            close= EXCLUDED.close, volume=EXCLUDED.volume,
            trade_count=EXCLUDED.trade_count, vwap=EXCLUDED.vwap;
        """
    )
    with engine.begin() as connection:
        # Efficient bulk operation
        connection.execute(sql, rows_list)
    return len(rows_list)


# ---------------------------------------------------------------------------
# Market data fetchers
# ---------------------------------------------------------------------------

def _process_bar_row(r: pd.Series, ts: pd.Timestamp, sym_override: str = None) -> Dict[str, Any]:
    """Helper function to process a Pandas Series row into the database dictionary format."""
    # Handle potential NaN values before casting
    trade_count = int(r["trade_count"]) if "trade_count" in r and pd.notna(r["trade_count"]) else None
    vwap = float(r["vwap"]) if "vwap" in r and pd.notna(r["vwap"]) else None

    # Determine the symbol: use override if provided (from index), else check the row itself
    if sym_override:
        symbol = str(sym_override)
    else:
        # If symbol is in the row (column), use it.
        symbol = str(r.get("symbol")) if pd.notna(r.get("symbol")) else None

    if not symbol:
        # Should be caught by upstream logic, but as a safeguard:
        print(f"[alpaca] Warning: Could not determine symbol for row at {ts}", file=sys.stderr)
        return None

    return {
        "symbol": symbol,
        "ts": ts.to_pydatetime().replace(tzinfo=None), # Ensure timezone-naive
        "open": float(r["open"]),
        "high": float(r["high"]),
        "low": float(r["low"]),
        "close": float(r["close"]),
        "volume": int(r.get("volume", 0) or 0),
        "trade_count": trade_count,
        "vwap": vwap,
    }

def _rows_from_alpaca(symbols: List[str], start: datetime, end: datetime) -> List[Dict[str, Any]]:
    """Fetch daily bars from Alpaca using alpaca-py, handling various response shapes."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("[alpaca] API credentials are missing; skipping Alpaca fetch", file=sys.stderr)
        return []

    client = StockHistoricalDataClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY)

    try:
        # Use Enums for configuration parameters
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment=Adjustment.RAW,
            feed=ALPACA_FEED_ENUM,
        )
        bars_response = client.get_stock_bars(request_params)
    except Exception as exc:
        print(f"[alpaca] error for symbols={symbols[:5]}...: {exc} (will fallback)", file=sys.stderr)
        return []

    if bars_response is None or bars_response.df is None or bars_response.df.empty:
        return []

    bars = bars_response.df
    rows: List[Dict[str, Any]] = []

    # Handle different DataFrame structures (Robust handling)
    if getattr(bars.index, "nlevels", 1) == 2 and set(bars.index.names) == {"symbol", "timestamp"}:
        # Multi-index path (multiple symbols requested)
        for (sym, ts), r in bars.iterrows():
            # Symbol is in the index
            processed_row = _process_bar_row(r, ts, sym_override=sym)
            if processed_row:
                rows.append(processed_row)

    elif "timestamp" in bars.index.names or isinstance(bars.index, pd.DatetimeIndex):
         # Single-index path (usually one symbol requested, index is the timestamp)
        # If symbol is not a column, try to infer it if only one was requested
        inferred_symbol = symbols[0] if len(symbols) == 1 and "symbol" not in bars.columns else None
        for ts, r in bars.iterrows():
            processed_row = _process_bar_row(r, ts, sym_override=inferred_symbol)
            if processed_row:
                rows.append(processed_row)
    else:
        print(f"[alpaca] Warning: Unrecognized DataFrame index structure: {bars.index.names}", file=sys.stderr)

    return rows


def _rows_from_tiingo(symbols: List[str], start: datetime, end: datetime) -> List[Dict[str, Any]]:
    # (Implementation remains the same as previously provided)
    if not TIINGO_API_KEY:
        return []
    rows: List[Dict[str, Any]] = []
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    for sym in symbols:
        url = (
            f"https://api.tiingo.com/tiingo/daily/{sym}/prices"
            f"?startDate={start_s}&endDate={end_s}&format=json"
            f"&resampleFreq=daily&columns=open,high,low,close,volume&token={TIINGO_API_KEY}"
        )
        try:
            resp = requests.get(url, headers={"Content-Type": "application/json"}, timeout=20)
            if resp.status_code == 404: continue
            resp.raise_for_status()
            data = resp.json() or []
            for entry in data:
                ts_val = datetime.fromisoformat(entry["date"].replace("Z", "+00:00")).replace(tzinfo=None)
                rows.append({
                    "symbol": sym, "ts": ts_val,
                    "open": float(entry.get("open", 0.0)), "high": float(entry.get("high", 0.0)),
                    "low": float(entry.get("low", 0.0)), "close": float(entry.get("close", 0.0)),
                    "volume": int(entry.get("volume", 0) or 0),
                    "trade_count": None, "vwap": None,
                })
        except Exception as exc:
            print(f"[tiingo] {sym} error: {exc}", file=sys.stderr)
            time.sleep(1)
    return rows


# ---------------------------------------------------------------------------
# Ingestion orchestration
# ---------------------------------------------------------------------------

def fetch_and_store(symbols: List[str], start_date: datetime, end_date: datetime) -> int:
    print(f"[fetch_and_store] symbols={len(symbols)} feed={ALPACA_FEED_ENUM.value} start={start_date} end={end_date}")
    engine = _get_engine()
    _ensure_schema(engine)
    written = 0
    # Process in chunks of 100
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        alpaca_rows = _rows_from_alpaca(chunk, start_date, end_date)
        if alpaca_rows:
            n = _insert_rows(engine, alpaca_rows)
            written += n
            print(f"[alpaca] chunk {i // 100} wrote {n} rows (total {written})")
        else:
            tiingo_rows = _rows_from_tiingo(chunk, start_date, end_date)
            if tiingo_rows:
                n = _insert_rows(engine, tiingo_rows)
                written += n
                print(f"[tiingo] chunk {i // 100} wrote {n} rows (total {written})")
    return written


# ---------------------------------------------------------------------------
# HTTP Routes
# ---------------------------------------------------------------------------

@app.route("/ingest", methods=["POST"])
def ingest_daily_data():
    try:
        payload: Dict[str, Any] = request.get_json(silent=True) or {}
        days_param = payload.get("days")
        if days_param is not None:
            try:
                days = int(days_param)
            except (ValueError, TypeError):
                return jsonify({"status": "error", "error": "'days' must be an integer"}), 400
        else:
            days = int(os.environ.get("BACKFILL_DAYS", "365"))
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        count = fetch_and_store(SYMBOLS, start, end)
        return jsonify({
            "status": "ok",
            "rows_written": count,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "symbols": len(SYMBOLS),
            "feed": ALPACA_FEED_ENUM.value,
            "fallback": bool(TIINGO_API_KEY),
        }), 200
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/debug", methods=["GET"])
def debug_probe():
    symbol = request.args.get("symbol", "AAPL").upper()
    try:
        days = int(request.args.get("days", "5"))
    except ValueError:
        return jsonify({"error": "Invalid 'days' parameter"}), 400
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    info: Dict[str, Any] = {"symbol": symbol, "start": start.isoformat(), "end": end.isoformat()}
    rows = _rows_from_alpaca([symbol], start, end)
    info["alpaca_rows"] = len(rows)
    if not rows and TIINGO_API_KEY:
        rows = _rows_from_tiingo([symbol], start, end)
        info["tiingo_rows"] = len(rows)

    # Ensure sample data is serializable
    sample = []
    if rows:
        for row in rows[:3]:
            r = row.copy()
            if isinstance(r.get('ts'), datetime):
                r['ts'] = r['ts'].isoformat()
            sample.append(r)
    info["sample"] = sample

    return jsonify(info), 200


@app.route("/")
def health_check():
    return "Service is running.", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
    # For local debugging only; Render runs gunicorn
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
