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
    """Filter (symbol, ts) rows to those investable as of ts using snapshots. Preserves original row order."""
    if df.empty or 'symbol' not in df.columns or 'ts' not in df.columns:
        return df
    # Load snapshots up to max ts in df
    lo = pd.to_datetime(df['ts'].min()).date()
    hi = pd.to_datetime(df['ts'].max()).date()
    with engine.connect() as con:
        uh = pd.read_sql_query(text("""
            SELECT as_of, symbol FROM universe_history
            WHERE as_of <= :hi
        """), con, params={'hi': hi}, parse_dates=['as_of'])
    if uh.empty:
        return df  # no gating if snapshots missing
    uh['as_of'] = pd.to_datetime(uh['as_of']).dt.date

    keep_indices = []
    for s, g in df.groupby('symbol', sort=False):
        snaps = uh[uh['symbol'] == s].sort_values('as_of')[['as_of']]
        if snaps.empty:
            continue  # no snapshots for this symbol -> drop all rows for s
        ts_series = pd.to_datetime(g['ts']).dt.date.rename('tsd').to_frame()
        m = pd.merge_asof(ts_series.sort_values('tsd'),
                          snaps.rename(columns={'as_of': 'snap'}).sort_values('snap'),
                          left_on='tsd', right_on='snap', direction='backward')
        investable = m['snap'].notna().values
        # collect original indices where investable == True
        keep_indices.extend(g.index[investable].tolist())

    if not keep_indices:
        return df.iloc[0:0]  # none investable
    return df.loc[sorted(keep_indices)]  # preserve original order across symbols
