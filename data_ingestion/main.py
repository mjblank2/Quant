# Render-ready ingestion service: SQLAlchemy(psycopg3) + Alpaca v3 with Tiingo fallback
# Routes:
#   "/"       -> health
#   "/ingest" -> POST {"days": 30}   backfill (default 2y)
#   "/debug"  -> GET ?symbol=AAPL&days=5  quick probe (no DB write)

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Iterable, List, Dict, Any, Optional

import requests
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
ALPACA_FEED = os.environ.get("ALPACA_FEED", "iex").lower().strip() or "iex"

TIINGO_API_KEY = os.environ.get("TIINGO_API_KEY")

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
if not TIINGO_API_KEY:
    print("[boot] WARNING: TIINGO_API_KEY not set; fallback will be unavailable.", file=sys.stderr)

alpaca = REST(key_id=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

# --------------------
# DB helpers (psycopg3)
# --------------------
def _psycopg3_url(url: str) -> str:
    if not url:
        raise RuntimeError("DATABASE_URL env var is required")
    if url.startswith("postgresql+psycopg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    return url

def _get_engine() -> Engine:
    return create_engine(_psycopg3_url(DATABASE_URL), pool_pre_ping=True)

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
    );"""
    with engine.begin() as conn:
        conn.exec_driver_sql(ddl)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
def _insert_rows(engine: Engine, rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    sql = text("""
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
    """)
    with engine.begin() as conn:
        for r in rows:
            conn.execute(sql, r)
    return len(rows)

# --------------------
# Data fetchers
# --------------------
def _rows_from_alpaca(symbols: List[str], start: datetime, end: datetime) -> List[Dict[str, Any]]:
    """Fetch daily bars from Alpaca; on any error, log and return [] so callers can fallback."""
    try:
        # Alpaca daily bars prefer date-only strings (RFC3339 date)
        start_s = start.date().isoformat()
        end_s = end.date().isoformat()

        bars = alpaca.get_bars(
            symbols,
            TimeFrame.Day,
            start=start_s,
            end=end_s,
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

def _rows_from_tiingo(symbols: List[str], start: datetime, end: datetime) -> List[Dict[str, Any]]:
    """Fetch daily bars from Tiingo (IEX base) for each symbol."""
    if not TIINGO_API_KEY:
        return []
    headers = {"Content-Type": "application/json"}
    rows: List[Dict[str, Any]] = []
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    for sym in symbols:
        url = (
            f"https://api.tiingo.com/tiingo/daily/{sym}/prices"
            f"?startDate={start_s}&endDate={end_s}&format=json&resampleFreq=daily&columns=open,high,low,close,volume"
            f"&token={TIINGO_API_KEY}"
        )
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json() or []
            for d in data:
                ts = datetime.fromisoformat(d["date"].replace("Z", "+00:00")).replace(tzinfo=None)
                rows.append({
                    "symbol": sym,
                    "ts": ts,
                    "open": float(d.get("open", 0.0)),
                    "high": float(d.get("high", 0.0)),
                    "low": float(d.get("low", 0.0)),
                    "close": float(d.get("close", 0.0)),
                    "volume": int(d.get("volume", 0) or 0),
                    "trade_count": None,
                    "vwap": None,
                })
        except Exception as e:
            print(f"[tiingo] {sym} error: {e}", file=sys.stderr)
            time.sleep(1)
    return rows

# --------------------
# Ingestion orchestration
# --------------------
def fetch_and_store(symbols: List[str], start_date: datetime, end_date: datetime) -> int:
    print(f"[fetch_and_store] symbols={len(symbols)} feed={ALPACA_FEED} start={start_date} end={end_date}")
    engine = _get_engine()
    _ensure_schema(engine)

    written = 0
    try:
        for i in range(0, len(symbols), 100):
            chunk = symbols[i:i+100]
            alpaca_rows = _rows_from_alpaca(chunk, start_date, end_date)
            if alpaca_rows:
                w = _insert_rows(engine, alpaca_rows)
                written += w
                print(f"[alpaca] chunk {i//100} wrote {w} rows (total {written})")
            else:
                tiingo_rows = _rows_from_tiingo(chunk, start_date, end_date)
                if tiingo_rows:
                    w = _insert_rows(engine, tiingo_rows)
                    written += w
                    print(f"[tiingo] chunk {i//100} wrote {w} rows (total {written})")
    except Exception as e:
        print(f"[fetch_and_store] FATAL: {e}", file=sys.stderr)

    return written

# --------------------
# HTTP Routes
# --------------------
@app.route("/ingest", methods=["POST"])
def ingest_daily_data():
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
            "fallback": bool(TIINGO_API_KEY),
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/debug", methods=["GET"])
def debug_probe():
    sym = request.args.get("symbol", "AAPL").upper()
    days = int(request.args.get("days", "5"))
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    info: Dict[str, Any] = {"symbol": sym, "start": start_date.isoformat(), "end": end_date.isoformat()}
    try:
        rows = _rows_from_alpaca([sym], start_date, end_date)
        info["alpaca_rows"] = len(rows)
        if not rows and TIINGO_API_KEY:
            rows = _rows_from_tiingo([sym], start_date, end_date)
            info["tiingo_rows"] = len(rows)
        info["sample"] = rows[:3] if rows else []
        return jsonify(info), 200
    except Exception as e:
        info["error"] = str(e)
        return jsonify(info), 200

@app.route("/")
def health():
    return "Service is running.", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))



