from __future__ import annotations
from datetime import date
import pandas as pd
from sqlalchemy import text
from db import get_engine, upsert_dataframe, DailyBar
from config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, ALPACA_DATA_FEED, POLYGON_API_KEY
from utils_http import get_json

POLY_BASE = "https://api.polygon.io"

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
        use = ["symbol","ts","open","high","low","close","adj_close","volume","vwap","trade_count"]
        return df[use]
    except Exception:
        return pd.DataFrame()

def _fetch_polygon_daily(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    url = f"{POLY_BASE}/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    js = get_json(url, {"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY, "limit": 50000}, timeout=30, max_tries=5)
    rows = (js or {}).get("results") or []
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["t"], unit="ms").dt.date
    df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume","vw":"vwap","n":"trade_count"}, inplace=True)
    df["adj_close"] = df["close"]
    return df[["ts","open","high","low","close","adj_close","volume","vwap","trade_count"]]

def ingest_bars_for_universe(days: int = 365) -> None:
    eng = get_engine()
    uni = pd.read_sql_query(text("SELECT symbol FROM universe WHERE included = TRUE"), eng)
    symbols = uni['symbol'].tolist()
    if not symbols:
        return

    end = pd.Timestamp('today').normalize().date()
    start = (pd.Timestamp(end) - pd.Timedelta(days=int(days*1.2))).date()
    start_s, end_s = pd.Timestamp(start).strftime('%Y-%m-%d'), pd.Timestamp(end).strftime('%Y-%m-%d')

    fetched = set()
    # Alpaca batch
    for i in range(0, len(symbols), 300):
        subs = symbols[i:i+300]
        df = _bars_from_alpaca_batch(subs, start, end)
        if not df.empty:
            upsert_dataframe(df, DailyBar, ['symbol','ts'])
            fetched |= set(df['symbol'].unique().tolist())

    remaining = sorted(set(symbols) - fetched)
    all_rows = []
    for sym in remaining:
        df = _fetch_polygon_daily(sym, start_s, end_s)
        if df is not None and not df.empty:
            df["symbol"] = sym
            all_rows.append(df)

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=['symbol','ts'])
        upsert_dataframe(out, DailyBar, ['symbol','ts'])

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=365)
    args = p.parse_args()
    ingest_bars_for_universe(args.days)

