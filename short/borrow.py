from __future__ import annotations
import os
import pandas as pd
from sqlalchemy import text
from db import engine

DDL = """
CREATE TABLE IF NOT EXISTS short_borrow (
  symbol VARCHAR(20) NOT NULL,
  ts DATE NOT NULL,
  available BOOLEAN NOT NULL,
  fee_bps FLOAT,
  short_interest FLOAT,
  source VARCHAR(32),
  PRIMARY KEY(symbol, ts)
);
CREATE INDEX IF NOT EXISTS ix_short_borrow_ts ON short_borrow(ts);
CREATE INDEX IF NOT EXISTS ix_short_borrow_symbol ON short_borrow(symbol);
"""

def _ensure():
    with engine.begin() as con:
        for stmt in DDL.strip().split(';'):
            s = stmt.strip()
            if s:
                con.execute(text(s))

def load_borrow_csv(path: str) -> int:
    """Load point-in-time borrow snapshots from CSV with columns:
    symbol, ts, available(0/1), fee_bps, short_interest(optional)
    """
    _ensure()
    import os
    if not path or not os.path.exists(path):
        return 0
    df = pd.read_csv(path)
    df['ts'] = pd.to_datetime(df['ts']).dt.date
    if 'available' in df.columns:
        df['available'] = df['available'].astype(int).astype(bool)
    if 'short_interest' not in df.columns:
        df['short_interest'] = None

    with engine.begin() as con:
        for _, r in df.iterrows():
            con.execute(text("""
                INSERT INTO short_borrow(symbol, ts, available, fee_bps, short_interest, source)
                VALUES (:s,:t,:a,:f,:si,'csv')
                ON CONFLICT (symbol, ts) DO UPDATE
                SET available=EXCLUDED.available,
                    fee_bps=EXCLUDED.fee_bps,
                    short_interest=EXCLUDED.short_interest,
                    source='csv'
            """), {'s': r['symbol'], 't': r['ts'], 'a': bool(r['available']), 'f': float(r.get('fee_bps') or 0.0), 'si': r.get('short_interest')})

    return int(len(df))

def borrow_state_asof(symbols: list[str], as_of) -> pd.DataFrame:
    """Returns availability and fee_bps for the most recent snapshot <= as_of for each symbol."""
    _ensure()
    if not symbols:
        return pd.DataFrame(columns=['symbol','available','fee_bps','short_interest'])
    as_of = pd.to_datetime(as_of).date()
    with engine.connect() as con:
        df = pd.read_sql_query(text("""
            SELECT s.symbol, s.ts, s.available, s.fee_bps, s.short_interest
            FROM short_borrow s
            WHERE s.symbol IN :syms AND s.ts <= :asof
            ORDER BY s.symbol, s.ts
        """), con, params={'syms': tuple(symbols), 'asof': as_of})
    if df.empty:
        return pd.DataFrame({'symbol': symbols, 'available': [False]*len(symbols), 'fee_bps':[None]*len(symbols), 'short_interest':[None]*len(symbols)})
    out_rows = []
    for sym in symbols:
        g = df[df['symbol']==sym]
        if g.empty:
            out_rows.append({'symbol': sym, 'available': False, 'fee_bps': None, 'short_interest': None})
        else:
            last = g.iloc[-1]
            out_rows.append({'symbol': sym, 'available': bool(last['available']), 'fee_bps': float(last['fee_bps']) if last['fee_bps'] is not None else None, 'short_interest': last.get('short_interest')})
    return pd.DataFrame(out_rows)

def borrow_fee_carry_daily(weight: float, fee_bps: float, equity: float) -> float:
    """Daily borrow carrying cost (negative number reduces PnL)."""
    if fee_bps is None:
        return 0.0
    ann = float(fee_bps) / 10000.0
    return -abs(weight) * equity * (ann / 252.0)