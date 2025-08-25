from __future__ import annotations
import pandas as pd
from datetime import date
from sqlalchemy import text
from db import engine, upsert_dataframe, UniverseHistory
from config import UNIVERSE_FILTER_RUSSELL, RUSSELL_INDEX

def take_snapshot(as_of: date | None = None) -> int:
    """Snapshot current investable set into universe_history for survivorship-safe training."""
    as_of = as_of or pd.Timestamp('today').normalize().date()
    with engine.connect() as con:
        uni = pd.read_sql_query(text("""
            SELECT symbol FROM universe WHERE included = TRUE ORDER BY symbol
        """), con)
    if uni.empty:
        return 0
    if UNIVERSE_FILTER_RUSSELL:
        # Use Russell membership table if present; otherwise keep all
        with engine.connect() as con:
            ru = pd.read_sql_query(text("""
                SELECT DISTINCT symbol FROM russell_membership
                WHERE ts <= :asof AND action IN ('add','keep')
            """), con, params={'asof': as_of})
        if not ru.empty:
            uni = uni[uni['symbol'].isin(ru['symbol'])]
    snap = uni.assign(as_of=as_of)[['as_of','symbol']]
    upsert_dataframe(snap, UniverseHistory, ['as_of','symbol'])
    return int(len(snap))

def gate_training_with_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Filter (symbol, ts) rows to those investable as of ts using snapshots."""
    if df.empty:
        return df
    # Load all snapshots covering df['ts'] range
    lo, hi = pd.to_datetime(df['ts'].min()).date(), pd.to_datetime(df['ts'].max()).date()
    with engine.connect() as con:
        uh = pd.read_sql_query(text("""
            SELECT as_of, symbol FROM universe_history
            WHERE as_of <= :hi
        """), con, params={'hi': hi}, parse_dates=['as_of'])
    if uh.empty:
        return df  # no gating if snapshots missing
    uh['as_of'] = pd.to_datetime(uh['as_of']).dt.date
    # For fast membership check, for each (symbol, ts) we need existence of snapshot <= ts
    # Build last snapshot date per symbol not exceeding ts via merge_asof per symbol
    out = []
    for s, g in df.groupby('symbol'):
        snaps = uh[uh['symbol']==s].sort_values('as_of')[['as_of']].rename(columns={'as_of':'snap'})
        if snaps.empty:
            continue
        gg = g.sort_values('ts').copy()
        gg['tsd'] = pd.to_datetime(gg['ts']).dt.date
        m = pd.merge_asof(gg[['tsd']], snaps, left_on='tsd', right_on='snap', direction='backward')
        gg['investable'] = m['snap'].notna()
        out.append(gg.assign(symbol=s))
    if not out:
        return df.iloc[0:0]
    inv = pd.concat(out, ignore_index=True)
    keep_idx = inv['investable'].values
    # Reindex back to the original order
    df2 = df.reset_index(drop=True).copy()
    df2 = df2.loc[keep_idx].copy()
    return df2
