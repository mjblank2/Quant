from __future__ import annotations
from datetime import date
import pandas as pd
from sqlalchemy import text
from db import engine, upsert_dataframe, DailyBar
from config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, ALPACA_DATA_FEED, TIINGO_API_KEY, POLYGON_API_KEY, HTTP_TIMEOUT
from utils_http import get_json

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
        df["adj_close"] = df["close"]  # adjustment='all'
        use = ["symbol","ts","open","high","low","close","adj_close","volume","vwap","trade_count"]
        return df[use]
    except Exception:
        return pd.DataFrame()

def _fetch_polygon_daily(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    if not POLYGON_API_KEY:
        return None
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    js = get_json(url, params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}, timeout=HTTP_TIMEOUT)
    if not js or not js.get("results"):
        return None
    rows = []
    for r in js["results"]:
        ts = pd.to_datetime(r.get("t"), unit="ms").date()
        rows.append({
            "ts": ts,
            "open": r.get("o"),
            "high": r.get("h"),
            "low": r.get("l"),
            "close": r.get("c"),
            "adj_close": r.get("c"),
            "volume": int(r.get("v") or 0),
            "vwap": r.get("vw"),
            "trade_count": r.get("n"),
            "symbol": symbol,
        })
    return pd.DataFrame(rows)

def _fetch_tiingo_daily(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    if not TIINGO_API_KEY:
        return None
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    js = get_json(url, params={'startDate': start, 'endDate': end, 'token': TIINGO_API_KEY}, timeout=HTTP_TIMEOUT)
    if not js:
        return None
    df = pd.DataFrame(js)
    if 'date' not in df.columns:
        return None
    df['ts'] = pd.to_datetime(df['date']).dt.date
    df['adj_close'] = df['adjClose'] if 'adjClose' in df.columns else df['close']
    out = df[['ts','open','high','low','close','adj_close','volume']].copy()
    out['symbol'] = symbol
    return out

def ingest_bars_for_universe(days: int = 365) -> None:
    uni = pd.read_sql_query(text("SELECT symbol FROM universe WHERE included = TRUE"), engine)
    symbols = uni['symbol'].tolist()
    if not symbols:
        return

    end = pd.Timestamp.utcnow().normalize().date()
    start = (pd.Timestamp(end) - pd.Timedelta(days=int(days*1.2))).date()
    start_s = pd.Timestamp(start).strftime('%Y-%m-%d')
    end_s = pd.Timestamp(end).strftime('%Y-%m-%d')

    # 1) Alpaca batch
    fetched = []
    for i in range(0, len(symbols), 300):
        subs = symbols[i:i+300]
        df = _bars_from_alpaca_batch(subs, start, end)
        if not df.empty:
            upsert_dataframe(df, DailyBar, ['symbol','ts'])
            fetched.extend(df['symbol'].unique().tolist())

    remaining = sorted(set(symbols) - set(fetched))

    # 2) Polygon per-symbol
    pol_rows = []
    for sym in remaining:
        df = _fetch_polygon_daily(sym, start_s, end_s)
        if df is not None and not df.empty:
            pol_rows.append(df[['symbol','ts','open','high','low','close','adj_close','volume','vwap','trade_count']])
    if pol_rows:
        out = pd.concat(pol_rows, ignore_index=True).drop_duplicates(subset=['symbol','ts'])
        upsert_dataframe(out, DailyBar, ['symbol','ts'])
        fetched.extend(out['symbol'].unique().tolist())

    remaining = sorted(set(symbols) - set(fetched))

    # 3) Tiingo fallback
    tia_rows = []
    for sym in remaining:
        df = _fetch_tiingo_daily(sym, start_s, end_s)
        if df is not None and not df.empty:
            df['vwap'] = None
            df['trade_count'] = None
            tia_rows.append(df[['symbol','ts','open','high','low','close','adj_close','volume','vwap','trade_count']])
    if tia_rows:
        out = pd.concat(tia_rows, ignore_index=True).drop_duplicates(subset=['symbol','ts'])
        upsert_dataframe(out, DailyBar, ['symbol','ts'])

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=365)
    args = p.parse_args()
    ingest_bars_for_universe(args.days)
