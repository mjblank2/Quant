from __future__ import annotations
import pandas as pd
from sqlalchemy import text
from db import engine, upsert_dataframe, ShortBorrow

def load_borrow_csv(path: str, source: str = "file") -> int:
    df = pd.read_csv(path)
    # Expected columns: ts,symbol,available,fee_bps,short_interest
    df['ts'] = pd.to_datetime(df['ts']).dt.date
    df['symbol'] = df['symbol'].str.upper().str.strip()
    df['source'] = source
    upsert_dataframe(df[['symbol','ts','available','fee_bps','short_interest','source']], ShortBorrow, ['symbol','ts'])
    return int(len(df))

def borrow_state_asof(symbols: list[str], as_of) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=['symbol','available','fee_bps','short_interest'])
    with engine.connect() as con:
        df = pd.read_sql_query(text("""
            SELECT symbol, ts, available, fee_bps, short_interest
            FROM short_borrow WHERE symbol IN :syms ORDER BY symbol, ts
        """), con, params={'syms': tuple(symbols)}, parse_dates=['ts'])
    if df.empty:
        return pd.DataFrame(columns=['symbol','available','fee_bps','short_interest'])
    as_of = pd.to_datetime(as_of)
    out = []
    for s, g in df.groupby('symbol'):
        g = g[g['ts'] <= as_of].sort_values('ts')
        if g.empty: continue
        last = g.iloc[-1]
        out.append({'symbol': s, 'available': last['available'], 'fee_bps': last['fee_bps'], 'short_interest': last['short_interest']})
    return pd.DataFrame(out)
