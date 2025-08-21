# data_ingestion/core.py
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Iterable, List, Dict, Any

import requests
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from tenacity import retry, stop_after_attempt, wait_exponential

# --- Config ---
ALPACA_API_KEY: str | None = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY: str | None = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://api.alpaca.markets").strip()
ALPACA_FEED: str = (os.getenv("ALPACA_FEED", "iex") or "iex").strip().lower()
TIINGO_API_KEY: str | None = os.getenv("TIINGO_API_KEY")
DATABASE_URL: str | None = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable must be set")

# Default universe if STOCK_UNIVERSE not set (comma‐separated symbols)
_default_symbols = ["AAPL", "MSFT", "SPY", "QQQ", "PLTR", "SMCI", "CRWD", "DDOG"]
SYMBOLS: List[str] = (
    [s.strip().upper() for s in os.getenv("STOCK_UNIVERSE", "").split(",") if s.strip()]
    or _default_symbols
)

def _psycopg3_url(url: str) -> str:
    if url.startswith("postgresql+psycopg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    return url

def get_engine() -> Engine:
    return create_engine(_psycopg3_url(DATABASE_URL), pool_pre_ping=True)

def ensure_schema(engine: Engine) -> None:
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
def insert_rows(engine: Engine, rows: Iterable[Dict[str, Any]]) -> int:
    rows_list = list(rows)
    if not rows_list:
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
    # Bulk executemany -> far fewer roundtrips than per-row executes
    with engine.begin() as conn:
        conn.execute(sql, rows_list)
    return len(rows_list)

def rows_from_tiingo(symbols: List[str], start: datetime, end: datetime) -> List[Dict[str, Any]]:
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
            for e in data:
                ts_val = datetime.fromisoformat(e["date"].replace("Z", "+00:00")).replace(tzinfo=None)
                rows.append({
                    "symbol": sym,
                    "ts": ts_val,
                    "open": float(e.get("open", 0.0)),
                    "high": float(e.get("high", 0.0)),
                    "low":  float(e.get("low", 0.0)),
                    "close": float(e.get("close", 0.0)),
                    "volume": int(e.get("volume", 0) or 0),
                    "trade_count": None,
                    "vwap": None,
                })
        except Exception as exc:
            print(f"[tiingo] {sym} error: {exc}", file=sys.stderr)
            time.sleep(1)
    return rows

# ---- Alpaca via alpaca-py (REST only; no streaming required) ----
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

def rows_from_alpaca(symbols: List[str], start: datetime, end: datetime) -> List[Dict[str, Any]]:
    if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
        print("[alpaca] API credentials missing; skipping", file=sys.stderr)
        return []
    try:
        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            adjustment="raw",
            feed=ALPACA_FEED,
        )
        resp = client.get_stock_bars(req)
        df: pd.DataFrame = resp.df  # MultiIndex (symbol, timestamp)
        if df is None or df.empty:
            return []
        rows: List[Dict[str, Any]] = []
        if getattr(df.index, "nlevels", 1) == 2:
            for (sym, ts), row in df.iterrows():
                rows.append({
                    "symbol": str(sym),
                    "ts": pd.to_datetime(ts).to_pydatetime().replace(tzinfo=None),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low":  float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row.get("volume", 0)),
                    "trade_count": int(row["trade_count"]) if "trade_count" in row and pd.notna(row["trade_count"]) else None,
                    "vwap": float(row["vwap"]) if "vwap" in row and pd.notna(row["vwap"]) else None,
                })
        else:
            if "symbol" not in df.columns:
                return []
            for _, row in df.iterrows():
                ts_val = row.get("timestamp", row.name)
                rows.append({
                    "symbol": str(row["symbol"]),
                    "ts": pd.to_datetime(ts_val).to_pydatetime().replace(tzinfo=None),
                    "open": float(row.get("open", 0.0)),
                    "high": float(row.get("high", 0.0)),
                    "low":  float(row.get("low", 0.0)),
                    "close": float(row.get("close", 0.0)),
                    "volume": int(row.get("volume", 0)),
                    "trade_count": int(row["trade_count"]) if "trade_count" in row and pd.notna(row["trade_count"]) else None,
                    "vwap": float(row["vwap"]) if "vwap" in row and pd.notna(row["vwap"]) else None,
                })
        return rows
    except Exception as exc:
        print(f"[alpaca] error for {symbols[:5]}...: {exc} (will fallback)", file=sys.stderr)
        return []

def fetch_and_store(symbols: List[str], start_date: datetime, end_date: datetime) -> int:
    print(f"[fetch_and_store] symbols={len(symbols)} feed={ALPACA_FEED} start={start_date} end={end_date}")
    engine = get_engine()
    ensure_schema(engine)
    written = 0
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        rows = rows_from_alpaca(chunk, start_date, end_date)
        if not rows:
            rows = rows_from_tiingo(chunk, start_date, end_date)
        if rows:
            n = insert_rows(engine, rows)
            written += n
            print(f"[ingest] chunk {i // 100} wrote {n} rows (total {written})")
    return written
