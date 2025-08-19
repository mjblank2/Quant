# Render-ready ingestion web service (psycopg3 + Alpaca v3).
# - "/"       → health check (200)
# - "/ingest" → POST {"days": 30} to backfill (defaults to 2 years)
# - Uses DATABASE_URL from Render; auto-creates daily_bars table if missing.

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Iterable, List

from flask import Flask, request, jsonify
from alpaca_trade_api.rest import REST, APIError, TimeFrame
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from tenacity import retry, stop_after_attempt, wait_exponential

app = Flask(__name__)

# --------------------
# Config from env
# --------------------
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://api.alpaca.markets")

# Market data feed: "iex" works on free/paper accounts; "sip" requires paid data
ALPACA_FEED = os.environ.get("ALPACA_FEED", "iex").lower().strip() or "iex"

DATABASE_URL = os.environ.get("DATABASE_URL")

_STOCKS = os.environ.get("STOCK_UNIVERSE")
SYMBOLS: List[str] = (
    [s.strip().upper() for s in _STOCKS.split(",") if s.strip()]
    if _STOCKS else
    ["AAPL", "MSFT", "SPY", "QQQ", "PLTR", "SMCI", "CRWD", "DDOG"]
)

# --------------------
# Clients
# --------------------
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    print("[boot] WARNING: ALPACA_API_KEY/ALPACA_SECRET_KEY not set.", file=sys.stderr)

api = REST(
    key_id=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
    base_url=ALPACA_BASE_URL,
)

# --------------------
# DB helpers (psycopg3)
# --------------------
def _psycopg3_url(url: str) -> str:
    """
    Ensure SQLAlchemy uses the psycopg3 driver.
    Render provides DATABASE_URL like 'postgresql://...'; switch to 'postgresql+psycopg://...'.
    """
    if not url:
        raise RuntimeError("DATABASE_URL env var is required")
    if url.startswith("postgresql+psycopg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    return url  # accept custom schemes

def _get_engine() -> Engine:
    dsn = _psycopg3_url(DATABASE_URL)
    eng = create_engine(dsn, pool_pre_ping=True)
    return eng

def _ensure_schema(engine: Engine) -> None:
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
    with engine.begin() as conn:
        conn.exec_driver_sql(ddl)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
def _rows_from_alpaca(symbols: List[str], start: datetime, end: datetime) -> List[Dict[str, Any]]:
    """Fetch daily bars from Alpaca; on any error, log and return [] so callers can fallback."""
    try:
        # Use DATE strings to satisfy Alpaca's RFC3339 expectation for daily bars
        start_s = start.date().isoformat()
        end_s = end.date().isoformat()

        bars = alpaca.get_bars(
            symbols,
            TimeFrame.Day,
            start=start_s,      # <-- date-only
            end=end_s,          # <-- date-only
            adjustment="raw",
            feed=ALPACA_FEED,
            limit=None,
        ).df
    except Exception as e:
        print(f"[alpaca] error for symbols={symbols[:5]}... : {e}  (will fallback to Tiingo)", file=sys.stderr)
        return []

    if bars is None or bars.empty:
        return []

    rows: List[Dict[str, Any]] = []
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

    if "symbol" not in bars.columns:
        return []

    for _, row in bars.iterrows():
        ts = row["timestamp"] if "timestamp" in row else row.name
        ts = ts.to_pydatetime().replace(tzinfo=None)
        rows.append({
            "symbol": str(row["symbol"]),
            "ts": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row.get("volume", 0)),
            "trade_count": int(row.get("trade_count", 0)) if "trade_count" in row else None,
            "vwap": float(row.get("vwap", 0.0)) if "vwap" in row else None,
        })
    return rows

# --------------------
# Ingestion
# --------------------
def fetch_and_store(symbols: List[str], start_date: datetime, end_date: datetime) -> int:
    """
    Fetch daily bars from Alpaca and store to Postgres.
    Returns number of rows written.
    """
    print(f"[fetch_and_store] symbols={len(symbols)} feed={ALPACA_FEED} start={start_date} end={end_date}")
    engine = _get_engine()
    _ensure_schema(engine)

    chunk_size = 100
    written = 0
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        try:
            bars_df = api.get_bars(
                chunk,
                TimeFrame.Day,
                start=start_date,
                end=end_date,
                adjustment="raw",
                feed=ALPACA_FEED,
                limit=None,
            ).df

            if bars_df is None or bars_df.empty:
                print(f"[fetch_and_store] Empty bars for chunk (n={len(chunk)}): {chunk[:5]}...")
                continue

            # bars_df index: MultiIndex(symbol, timestamp)
            rows = []
            # Guard against single-index (older clients) by checking .index.nlevels
            if getattr(bars_df.index, "nlevels", 1) == 2:
                for (sym, ts), row in bars_df.iterrows():
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
            else:
                # Fallback for single-index frames: require a 'symbol' column
                if "symbol" not in bars_df.columns:
                    print("[fetch_and_store] Unexpected bars format (missing symbol column); skipping chunk.")
                    continue
                for _, row in bars_df.iterrows():
                    ts = row["timestamp"] if "timestamp" in row else row.name
                    ts = ts.to_pydatetime().replace(tzinfo=None)
                    rows.append({
                        "symbol": str(row["symbol"]),
                        "ts": ts,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row.get("volume", 0)),
                        "trade_count": int(row.get("trade_count", 0)) if "trade_count" in row else None,
                        "vwap": float(row.get("vwap", 0.0)) if "vwap" in row else None,
                    })

            wrote = _insert_rows(engine, rows)
            written += wrote
            print(f"[fetch_and_store] chunk wrote {wrote} rows (total {written})")

        except APIError as e:
            print(f"[fetch_and_store] Alpaca APIError: {e}", file=sys.stderr)
            time.sleep(10)
        except Exception as e:
            print(f"[fetch_and_store] ERROR: {e}", file=sys.stderr)
            time.sleep(5)

    return written

# --------------------
# HTTP Routes
# --------------------
@app.route("/ingest", methods=["POST"])
def ingest_daily_data():
    """
    POST JSON: {"days": 365}
    Defaults to 2 years if not provided.
    """
    try:
        payload = request.get_json(silent=True) or {}
        days = int(payload.get("days", 365 * 2))
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        count = fetch_and_store(SYMBOLS, start_date, end_date)
        return jsonify({
            "status": "ok",
            "rows_written": count,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "symbols": len(SYMBOLS),
            "feed": ALPACA_FEED,
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/")
def health():
    return "Service is running.", 200

# --------------------
# Dev entrypoint (Render uses gunicorn)
# --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)


