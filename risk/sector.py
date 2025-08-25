from __future__ import annotations
import os, pandas as pd
from sqlalchemy import text
from db import engine

DDL = """
CREATE TABLE IF NOT EXISTS sector_map (
  symbol VARCHAR(20) NOT NULL,
  as_of DATE NOT NULL,
  sector VARCHAR(64),
  industry VARCHAR(128),
  PRIMARY KEY(symbol, as_of)
);
CREATE INDEX IF NOT EXISTS ix_sector_map_symbol ON sector_map(symbol);
CREATE INDEX IF NOT EXISTS ix_sector_map_asof ON sector_map(as_of);
"""

def _ensure():
    with engine.begin() as con:
        for stmt in DDL.strip().split(';'):
            s = stmt.strip()
            if s:
                con.execute(text(s))

def load_sector_map_csv(path: str) -> int:
    _ensure()
    if not path or not os.path.exists(path):
        return 0
    df = pd.read_csv(path)
    df['as_of'] = pd.to_datetime(df['as_of']).dt.date
    with engine.begin() as con:
        for _, r in df.iterrows():
            con.execute(text("""
                INSERT INTO sector_map(symbol, as_of, sector, industry)
                VALUES (:s,:a,:sec,:ind)
                ON CONFLICT (symbol, as_of) DO UPDATE
                SET sector=EXCLUDED.sector, industry=EXCLUDED.industry
            """), {'s': r['symbol'], 'a': r['as_of'], 'sec': r.get('sector'), 'ind': r.get('industry')})
    return int(len(df))

def sector_asof(symbols: list[str], as_of) -> pd.Series:
    _ensure()
    if not symbols:
        return pd.Series(dtype=object)
    with engine.connect() as con:
        df = pd.read_sql_query(text("""
            SELECT symbol, as_of, sector FROM sector_map WHERE symbol IN :syms ORDER BY symbol, as_of
        """), con, params={'syms': tuple(symbols)}, parse_dates=['as_of'])
    if df.empty:
        return pd.Series(index=symbols, dtype=object)
    as_of = pd.to_datetime(as_of).date()
    rows = []
    for s in symbols:
        g = df[df['symbol']==s]
        g = g[g['as_of']<=as_of].sort_values('as_of')
        rows.append(g.iloc[-1]['sector'] if not g.empty else None)
    return pd.Series(rows, index=symbols, dtype=object)

def build_sector_dummies(symbols: list[str], as_of) -> pd.DataFrame:
    _ensure()
    sec = sector_asof(symbols, as_of)
    if sec.empty:
        return pd.DataFrame(index=symbols)
    d = pd.get_dummies(sec.fillna("UNKNOWN"))
    d = d - d.mean(axis=0)  # sum-to-zero encoding
    d.index = symbols
    return d
