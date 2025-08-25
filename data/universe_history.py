from __future__ import annotations
from datetime import date
import pandas as pd
from sqlalchemy import text
from db import engine

DDL = """
CREATE TABLE IF NOT EXISTS universe_history (
  as_of DATE NOT NULL,
  symbol VARCHAR(20) NOT NULL,
  included BOOLEAN NOT NULL DEFAULT TRUE,
  PRIMARY KEY(as_of, symbol)
);
CREATE INDEX IF NOT EXISTS ix_universe_history_asof ON universe_history(as_of);
CREATE INDEX IF NOT EXISTS ix_universe_history_symbol ON universe_history(symbol);
"""

def _ensure():
    with engine.begin() as con:
        for stmt in DDL.strip().split(';'):
            s = stmt.strip()
            if s:
                con.execute(text(s))

def snapshot_universe(as_of: date | None = None) -> int:
    """Store a point-in-time snapshot of the current investable universe."""
    _ensure()
    if as_of is None:
        as_of = pd.Timestamp.today().normalize().date()
    with engine.connect() as con:
        uni = pd.read_sql_query(text("SELECT symbol FROM universe WHERE included=TRUE"), con)
    if uni.empty:
        return 0
    syms = uni['symbol'].tolist()
    with engine.begin() as con:
        for s in syms:
            con.execute(text("""
                INSERT INTO universe_history(as_of, symbol, included)
                VALUES (:d, :s, TRUE)
                ON CONFLICT (as_of, symbol) DO UPDATE SET included=TRUE
            """), {'d': as_of, 's': s})
    return len(syms)

def gate_training_with_universe(train_df: pd.DataFrame) -> pd.DataFrame:
    """Filter (symbol, ts) in train_df to those present in the universe snapshot as of ts (weekly snapshots)."""
    _ensure()
    if train_df is None or train_df.empty or 'symbol' not in train_df.columns or 'ts' not in train_df.columns:
        return train_df
    min_d, max_d = train_df['ts'].min(), train_df['ts'].max()
    with engine.connect() as con:
        uh = pd.read_sql_query(text("""
            SELECT as_of, symbol FROM universe_history
            WHERE as_of <= :mx
            ORDER BY symbol, as_of
        """), con, params={'mx': max_d})
    if uh.empty:
        return train_df
    # Merge-asof by symbol: for each (symbol, ts) find last snapshot <= ts
    df = train_df.copy()
    df['ts_key'] = pd.to_datetime(df['ts'])
    uh2 = uh.sort_values(['symbol','as_of']).copy()
    uh2['as_of'] = pd.to_datetime(uh2['as_of'])
    out_parts = []
    for sym, g in df.groupby('symbol'):
        snaps = uh2[uh2['symbol']==sym][['as_of']].rename(columns={'as_of':'snap'})
        if snaps.empty:
            continue
        gg = g.sort_values('ts_key').copy()
        gg = pd.merge_asof(gg, snaps, left_on='ts_key', right_on='snap', direction='backward')
        gg = gg[~gg['snap'].isna()]  # keep rows that have a snapshot
        out_parts.append(gg.drop(columns=['ts_key','snap']))
    if not out_parts:
        return train_df.iloc[0:0]
    return pd.concat(out_parts, ignore_index=True)