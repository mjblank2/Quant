"""
Main entrypoint for the data ingestion service.

This module defines a Flask application that exposes endpoints to backfill
historical daily bars for a universe of equity symbols and persist them
into a Postgres database.  It uses the alpaca‑trade‑api library to fetch
market data from the Alpaca API and automatically falls back to Tiingo
if Alpaca denies access (for example, when the account lacks market data
entitlements).

The module also defines helpers for working with SQLAlchemy and psycopg3,
including automatic schema creation and UPSERT logic.  The ingestion
routine is designed to be idempotent and safe to run repeatedly: rows
are inserted or updated on conflict, and a configurable lookback window
controls how far back to fetch.

To test locally:

    export DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
    export ALPACA_API_KEY=<your_key>
    export ALPACA_SECRET_KEY=<your_secret>
    export TIINGO_API_KEY=<optional_tiingo_key>
    python -m data_ingestion.main

On Render the environment variables are set via the dashboard or defined
in render.yaml.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Iterable, List, Dict, Any

import requests
from flask import Flask, request, jsonify
from alpaca_trade_api.rest import REST, APIError, TimeFrame
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from tenacity import retry, stop_after_attempt, wait_exponential

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

# API credentials for Alpaca.  These must be provided; if missing, the
# application will log a warning and Alpaca calls will fail.
ALPACA_API_KEY: str | None = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY: str | None = os.environ.get("ALPACA_SECRET_KEY")

# Base URL for the Alpaca API.  Use the live trading endpoint by default.
ALPACA_BASE_URL: str = os.environ.get("ALPACA_BASE_URL", "https://api.alpaca.markets").strip()

# Market data feed.  Free accounts typically use "iex"; paid plans may use "sip".
ALPACA_FEED: str = os.environ.get("ALPACA_FEED", "iex").strip().lower() or "iex"

# API token for Tiingo.  Used as a fallback when Alpaca refuses access.
TIINGO_API_KEY: str | None = os.environ.get("TIINGO_API_KEY")

# Database URL injected by Render or set locally.  Must be a Postgres DSN.
DATABASE_URL: str | None = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable must be set")

# Universe of symbols to backfill.  Can be overridden via STOCK_UNIVERSE env.
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
    """Ensure SQLAlchemy uses the psycopg3 driver.

    SQLAlchemy defaults to psycopg2 when the scheme is ``postgresql://``.
    We convert it to ``postgresql+psycopg://`` unless the URL already
    specifies a driver.

    Args:
        url: The original database URL.

    Returns:
        A URL with an explicit psycopg driver.
    """
    if url.startswith("postgresql+psycopg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def _get_engine() -> Engine:
    """Create a SQLAlchemy engine using psycopg3 and connection pooling."""
    dsn = _psycopg3_url(DATABASE_URL)
    return create_engine(dsn, pool_pre_ping=True)


def _ensure_schema(engine: Engine) -> None:
    """Create the `daily_bars` table if it does not exist."""
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
    """
    with engine.begin() as connection:
        connection.exec_driver_sql(ddl)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
def _insert_rows(engine: Engine, rows: Iterable[Dict[str, Any]]) -> int:
    """Insert or update rows into the daily_bars table.

    Args:
        engine: The SQLAlchemy engine.
        rows: An iterable of dictionaries mapping column names to values.

    Returns:
        The number of rows processed.
    """
    rows_list = list(rows)
    if not rows_list:
        return 0
    sql = text(
        """
        INSERT INTO daily_bars (symbol, ts, open, high, low, close, volume, trade_count, vwap)
        VALUES (:symbol, :ts, :open, :high, :low, :close, :volume, :trade_count, :vwap)
        ON CONFLICT (symbol, ts) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low  = EXCLUDED.low,
            close= EXCLUDED.close,
            volume=EXCLUDED.volume,
            trade_count=EXCLUDED.trade_count,
            vwap=EXCLUDED.vwap;
        """
    )
    with engine.begin() as connection:
        for row in rows_list:
            connection.execute(sql, row)
    return len(rows_list)


# ---------------------------------------------------------------------------
# Market data fetchers
# ---------------------------------------------------------------------------

def _rows_from_alpaca(symbols: List[str], start: datetime, end: datetime) -> List[Dict[str, Any]]:
    """Fetch daily bars from Alpaca.

    On any error (e.g., entitlement issues), the function logs the error
    and returns an empty list so the caller can fall back to another data
    source.

    Args:
        symbols: A list of symbols to fetch.
        start: The start of the date range.
        end: The end of the date range.

    Returns:
        A list of dictionaries representing bar rows.
    """
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("[alpaca] API credentials are missing; skipping Alpaca fetch", file=sys.stderr)
        return []
    client = REST(key_id=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)
    try:
        # Alpaca expects date strings (RFC 3339 date) for daily bars
        start_s = start.date().isoformat()
        end_s = end.date().isoformat()
        bars = client.get_bars(
            symbols,
            TimeFrame.Day,
            start=start_s,
            end=end_s,
            adjustment="raw",
            feed=ALPACA_FEED,
            limit=None,
        ).df
    except Exception as exc:
        print(f"[alpaca] error for symbols={symbols[:5]}...: {exc} (will fallback)", file=sys.stderr)
        return []
    if bars is None or bars.empty:
        return []
    rows: List[Dict[str, Any]] = []
    # MultiIndex (symbol, timestamp) expected
    if getattr(bars.index, "nlevels", 1) == 2:
        for (sym, ts), row in bars.iterrows():
            rows.append({
                "symbol": str(sym),
                "ts": ts.to_pydatetime().replace(tzinfo=None),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row.get("volume", 0)),
                "trade_count": int(row.get("trade_count", 0)) if "trade_count" in row else None,
                "vwap": float(row.get("vwap", 0.0)) if "vwap" in row else None,
            })
        return rows
    # Fallback: single‑index DataFrame
    if "symbol" not in bars.columns:
        return []
    for _, row in bars.iterrows():
        ts_val = row.get("timestamp", row.name)
        ts_val = ts_val.to_pydatetime().replace(tzinfo=None) if hasattr(ts_val, "to_pydatetime") else ts_val
        rows.append({
            "symbol": str(row.get("symbol")),
            "ts": ts_val,
            "open": float(row.get("open", 0.0)),
            "high": float(row.get("high", 0.0)),
            "low": float(row.get("low", 0.0)),
            "close": float(row.get("close", 0.0)),
            "volume": int(row.get("volume", 0)),
            "trade_count": int(row.get("trade_count", 0)) if "trade_count" in row else None,
            "vwap": float(row.get("vwap", 0.0)) if "vwap" in row else None,
        })
    return rows


def _rows_from_tiingo(symbols: List[str], start: datetime, end: datetime) -> List[Dict[str, Any]]:
    """Fetch daily bars from Tiingo for each symbol.

    Args:
        symbols: List of symbols to fetch.
        start: Start datetime.
        end: End datetime.

    Returns:
        A list of dictionaries representing bar rows.
    """
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
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json() or []
            for entry in data:
                # Tiingo returns 'date' like '2025-08-19T00:00:00.000Z'
                ts_val = datetime.fromisoformat(entry["date"].replace("Z", "+00:00")).replace(tzinfo=None)
                rows.append({
                    "symbol": sym,
                    "ts": ts_val,
                    "open": float(entry.get("open", 0.0)),
                    "high": float(entry.get("high", 0.0)),
                    "low": float(entry.get("low", 0.0)),
                    "close": float(entry.get("close", 0.0)),
                    "volume": int(entry.get("volume", 0) or 0),
                    "trade_count": None,
                    "vwap": None,
                })
        except Exception as exc:
            print(f"[tiingo] {sym} error: {exc}", file=sys.stderr)
            time.sleep(1)
    return rows


# ---------------------------------------------------------------------------
# Ingestion orchestration
# ---------------------------------------------------------------------------

def fetch_and_store(symbols: List[str], start_date: datetime, end_date: datetime) -> int:
    """Fetch bars for all symbols between start_date and end_date and upsert them.

    Args:
        symbols: The list of symbols to fetch.
        start_date: The beginning of the backfill window.
        end_date: The end of the backfill window.

    Returns:
        The number of rows written to the database.
    """
    print(f"[fetch_and_store] symbols={len(symbols)} feed={ALPACA_FEED} start={start_date} end={end_date}")
    engine = _get_engine()
    _ensure_schema(engine)
    written = 0
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        alpaca_rows = _rows_from_alpaca(chunk, start_date, end_date)
        if alpaca_rows:
            n = _insert_rows(engine, alpaca_rows)
            written += n
            print(f"[alpaca] chunk {i // 100} wrote {n} rows (total {written})")
        else:
            # Fallback to Tiingo only if available
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
    """Trigger a backfill for a specified number of days.

    The client can POST JSON with a ``days`` integer to override
    ``BACKFILL_DAYS``.  If omitted, ``BACKFILL_DAYS`` controls the
    lookback window.  The endpoint returns a JSON response with the
    number of rows written and the date range processed.
    """
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
            "feed": ALPACA_FEED,
            "fallback": bool(TIINGO_API_KEY),
        }), 200
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/debug", methods=["GET"])
def debug_probe():
    """Probe a single symbol over a short window and return a sample of rows.

    Query parameters:

    - ``symbol``: The symbol to fetch (default "AAPL").
    - ``days``: Number of days to look back (default 5).
    """
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
    info["sample"] = rows[:3] if rows else []
    return jsonify(info), 200


@app.route("/")
def health_check():
    """Simple health check endpoint."""
    return "Service is running.", 200


if __name__ == "__main__":
    # For local debugging only; Render runs gunicorn
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
