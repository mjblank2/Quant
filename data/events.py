from __future__ import annotations
import pandas as pd
from sqlalchemy import text
from db import engine, upsert_dataframe, AltSignal
from datetime import timedelta

DDL_RUSSELL = """
CREATE TABLE IF NOT EXISTS russell_membership (
  symbol VARCHAR(20) NOT NULL,
  as_of DATE NOT NULL,
  index_name VARCHAR(16) NOT NULL,  -- e.g., R2000, R3000
  member BOOLEAN NOT NULL,
  PRIMARY KEY(symbol, as_of, index_name)
);
CREATE INDEX IF NOT EXISTS ix_russell_asof ON russell_membership(as_of);
CREATE INDEX IF NOT EXISTS ix_russell_symbol ON russell_membership(symbol);
"""

def _ensure():
    with engine.begin() as con:
        for stmt in DDL_RUSSELL.strip().split(';'):
            s = stmt.strip()
            if s:
                con.execute(text(s))

def load_earnings_csv(csv_path: str) -> int:
    """Columns expected: symbol, announce_date, surprise_eps, surprise_rev (optional)

       Writes AltSignals: 'pead_surprise_eps' / 'pead_surprise_rev' at ts=announce_date+1 (T+1 tradeable)."""
    if not csv_path: return 0
    df = pd.read_csv(csv_path)
    if df.empty: return 0
    df['announce_date'] = pd.to_datetime(df['announce_date']).dt.date
    rows = []
    for _, r in df.iterrows():
        t = r['announce_date']
        t_trade = (pd.Timestamp(t) + timedelta(days=1)).date()
        if 'surprise_eps' in r and pd.notna(r['surprise_eps']):
            rows.append({'symbol': r['symbol'], 'ts': t_trade, 'name':'pead_surprise_eps', 'value': float(r['surprise_eps'])})
        if 'surprise_rev' in r and pd.notna(r.get('surprise_rev')):
            rows.append({'symbol': r['symbol'], 'ts': t_trade, 'name':'pead_surprise_rev', 'value': float(r['surprise_rev'])})
        # event day indicator (sparse)
        rows.append({'symbol': r['symbol'], 'ts': t_trade, 'name':'pead_event', 'value': 1.0})
    if not rows: return 0
    adf = pd.DataFrame(rows)
    upsert_dataframe(adf[['symbol','ts','name','value']], AltSignal, ['symbol','ts','name'])
    return len(adf)

def load_russell_membership_csv(csv_path: str) -> int:
    """Columns expected: symbol, as_of, index_name (R2000|R3000), member (1/0)

       Also emits AltSignals 'russell_inout' (+1 on join, -1 on drop) at that as_of date."""
    _ensure()
    if not csv_path: return 0
    df = pd.read_csv(csv_path)
    if df.empty: return 0
    df['as_of'] = pd.to_datetime(df['as_of']).dt.date
    # Upsert membership
    with engine.begin() as con:
        for _, r in df.iterrows():
            con.execute(text("""
                INSERT INTO russell_membership(symbol, as_of, index_name, member)
                VALUES (:s,:a,:i,:m)
                ON CONFLICT(symbol, as_of, index_name) DO UPDATE SET member=EXCLUDED.member
            """), {'s': r['symbol'], 'a': r['as_of'], 'i': r['index_name'], 'm': bool(int(r['member']))})

    # Create in/out AltSignals
    # Assumes rows are chronological per (symbol, index_name)
    df = df.sort_values(['symbol','index_name','as_of'])
    rows = []
    for (sym, idx), g in df.groupby(['symbol','index_name']):
        prev = None
        for _, r in g.iterrows():
            if prev is not None and prev != r['member']:
                rows.append({'symbol': sym, 'ts': r['as_of'], 'name':'russell_inout', 'value': 1.0 if r['member'] else -1.0})
            prev = r['member']
    if rows:
        adf = pd.DataFrame(rows)
        upsert_dataframe(adf[['symbol','ts','name','value']], AltSignal, ['symbol','ts','name'])
    return len(df)