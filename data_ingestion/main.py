# Render-ready ingestion web service (psycopg3).
# - Flask app exposes "/" (health) and "/ingest" (POST) to backfill data
# - Uses Alpaca for market data
# - Writes OHLCV to Postgres via SQLAlchemy + psycopg3 (psycopg[binary])
import os
import time
from datetime import datetime, timedelta
from typing import Iterable, List

from flask import Flask, request, jsonify
from alpaca_trade_api.rest import REST, APIError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from tenacity import retry, stop_after_attempt, wait_exponential

app = Flask(__name__)

# --- Config ---
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
DATABASE_URL = os.environ.get("DATABASE_URL")

# Universe can be overridden via env STOCK_UNIVERSE="AAPL,MSFT,GOOG"
_STOCKS = os.environ.get("STOCK_UNIVERSE")
SYMBOLS: List[str] = (
    [s.strip().upper() for s in _STOCKS.split(",") if s.strip()]
    if _STOCKS else
    ["SMCI","CRWD","DDOG","MDB","OKTA","PLTR","SNOW","ZS","ETSY","PINS","ROKU","SQ","TDOC","TWLO","U","ZM"]
)

# --- Clients ---
api = REST(key_id=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

def _psycopg3_url(url: str) -> str:
    """
    Ensure SQLAlchemy uses the psycopg3 driver.
    Render provides DATABASE_URL like 'postgresql://...'
    We switch to 'postgresql+psycopg://...' unless it's already set.
    """
    if not url:
        raise RuntimeError("DATABASE_URL env var is required")
    if url.startswith("postgresql+psycopg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    # Accept custom schemes (e.g., when proxied); fallback to given URL
    return url

def _get_engine() -> Engine:
    eng = create_engine(_psycopg3_url(DATABASE_URL), pool_pre_ping=True)
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
    total = 0
    with engine.begin() as conn:
        for r in rows:
            conn.execute(sql, r)
            total += 1
    return total

def fetch_and_store(symbols: List[str], start_date: datetime, end_date: datetime) -> int:
    """
    Fetch daily bars from Alpaca for symbols and store to Postgres.
    Returns number of rows written.
    """
    engine = _get_engine()
    _ensure_schema(engine)

    chunk_size = 100
    written = 0
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i+chunk_size]
        try:
            bars = api.get_bars(
                chunk, "1Day",
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d")
            ).df

            if bars.empty:
                continue

            # Index is MultiIndex(symbol, timestamp) in newer SDKs
            rows = []
            for (sym, ts), row in bars.iterrows():
                rows.append({
                    "symbol": sym,
                    "ts": ts.to_pydatetime().replace(tzinfo=None),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row.get("volume", 0)),
                    "trade_count": int(row.get("trade_count", 0)) if "trade_count" in row else None,
                    "vwap": float(row.get("vwap", 0.0)) if "vwap" in row else None,
                })
            written += _insert_rows(engine, rows)

        except APIError:
            time.sleep(10)
        except Exception:
            time.sleep(5)

    return written

@app.route("/ingest", methods=["POST"])
def ingest_daily_data():
    """
    POST JSON: {"days": 365}  (defaults to 2 years)
    """
    try:
        payload = request.get_json(silent=True) or {}
        days = int(payload.get("days", 365*2))
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        count = fetch_and_store(SYMBOLS, start_date, end_date)
        return jsonify({
            "status": "ok",
            "rows_written": count,
            "start": start_date.isoformat(),
            "end": end_date.isoformat()
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/")
def health():
    return "Service is running.", 200

if __name__ == "__main__":
    # Local dev only; on Render we use gunicorn
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))

