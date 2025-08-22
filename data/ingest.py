from __future__ import annotations
import requests
from datetime import date
import pandas as pd
from sqlalchemy import text
from db import engine, upsert_dataframe, DailyBar
from config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, ALPACA_DATA_FEED, TIINGO_API_KEY

def _date(s) -> str:
    return pd.Timestamp(s).strftime('%Y-%m-%d')

def _fetch_alpaca_daily(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        import alpaca_trade_api as tradeapi
        from alpaca_trade_api.rest import TimeFrame
        api = tradeapi.REST(key_id=APCA_API_KEY_ID, secret_key=APCA_API_SECRET_KEY, base_url=APCA_API_BASE_URL)
        bars = api.get_bars(symbol, TimeFrame.Day, start, end, adjustment='all', feed=ALPACA_DATA_FEED)
        df = bars.df if hasattr(bars, 'df') else None
        if df is None or df.empty:
            return None
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        df = df[['open','high','low','close','volume','vwap','trade_count']].copy()
        df = df.reset_index()
        if "timestamp" in df.columns:
        df.rename(columns={"timestamp": "ts"}, inplace=True)
        # set ts from the 'ts' column you just created (or from the index if unnamed)
        if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"]).dt.date
        else:
        df["ts"] = pd.to_datetime(df.index).date
        df.drop(columns=[c for c in ["index","timestamp"] if c in df.columns], inplace=True)
        df['adj_close'] = df['close']  # 'close' already adjusted when adjustment='all'
        return df
    except Exception:
        return None

def _fetch_tiingo_daily(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    if not TIINGO_API_KEY:
        return None
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    try:
        r = requests.get(url, params={'startDate': start, 'endDate': end, 'token': TIINGO_API_KEY}, timeout=20)
        if r.status_code != 200:
            return None
        js = r.json()
        if not js:
            return None
        df = pd.DataFrame(js)
        if 'date' not in df.columns:
            return None
        df['ts'] = pd.to_datetime(df['date']).dt.date
        # Some responses include 'adjClose'
        df.rename(columns={'open':'open','high':'high','low':'low','close':'close','volume':'volume'}, inplace=True)
        df['adj_close'] = df['adjClose'] if 'adjClose' in df.columns else df['close']
        return df[['ts','open','high','low','close','adj_close','volume']]
    except Exception:
        return None

def _fetch_yf_daily(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        raw = t.history(start=start, end=end, interval='1d', auto_adjust=False)
        adj = t.history(start=start, end=end, interval='1d', auto_adjust=True)
        if raw is None or raw.empty:
            return None
        if adj is None or adj.empty:
            adj = raw.copy()
        df = raw[['Open','High','Low','Close','Volume']].copy()
        df['AdjClose'] = adj['Close']
        df.reset_index(inplace=True)
        df.rename(columns={'Date':'ts','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume','AdjClose':'adj_close'}, inplace=True)
        df['ts'] = pd.to_datetime(df['ts']).dt.date
        return df[['ts','open','high','low','close','adj_close','volume']]
    except Exception:
        return None

def _fetch_daily(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    for fn in (_fetch_alpaca_daily, _fetch_tiingo_daily, _fetch_yf_daily):
        df = fn(symbol, start, end)
        if df is not None and not df.empty:
            df['symbol'] = symbol
            # optional NA cleanup
            df = df.dropna(subset=['ts','close'])
            return df
    return None

def ingest_bars_for_universe(days: int = 365) -> None:
    # Universe symbols
    uni = pd.read_sql_query(text("SELECT symbol FROM universe WHERE included = TRUE"), engine)
    symbols = uni['symbol'].tolist()

    if not symbols:
        return

    end = pd.Timestamp('today').normalize().date()
    start = (pd.Timestamp(end) - pd.Timedelta(days=int(days*1.2))).date()  # pad
    start_s, end_s = _date(start), _date(end)

    all_rows = []
    for sym in symbols:
        df = _fetch_daily(sym, start_s, end_s)
        if df is None or df.empty:
            continue
        all_rows.append(df[['symbol','ts','open','high','low','close','adj_close','volume']])

    if not all_rows:
        return

    out = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=['symbol','ts'])
    # vwap/trade_count optional from providers; just ensure columns exist
    out['vwap'] = None
    out['trade_count'] = None
    upsert_dataframe(out, DailyBar, ['symbol','ts'])
