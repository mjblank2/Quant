from __future__ import annotations
import numpy as np
import pandas as pd

def time_groups(dates: pd.Series, n_groups: int) -> pd.Series:
    """Label each date with a group id in order of time (roughly equal-sized)."""
    uniq = sorted(pd.to_datetime(pd.Series(dates).unique()))
    k = max(1, n_groups)
    bins = np.array_split(uniq, k)
    mapping = {}
    for gid, arr in enumerate(bins):
        for d in arr:
            mapping[pd.Timestamp(d)] = gid
    return pd.Series(pd.to_datetime(dates)).map(mapping)

def cpcv_splits(df: pd.DataFrame, date_col: str, n_groups: int = 5, embargo: int = 5):
    """Yield (train_idx, test_idx) with purging/embargo between group folds."""
    g = time_groups(df[date_col], n_groups=n_groups)
    df = df.copy()
    df['__g__'] = g.values
    for test_gid in sorted(df['__g__'].unique()):
        test_mask = (df['__g__'] == test_gid)
        test_dates = sorted(pd.to_datetime(df.loc[test_mask, date_col]).unique())
        if not test_dates:
            continue
        lo = test_dates[0] - pd.Timedelta(days=embargo)
        hi = test_dates[-1] + pd.Timedelta(days=embargo)
        train_mask = ~((df[date_col] >= lo) & (df[date_col] <= hi))
        yield df.index[train_mask].tolist(), df.index[test_mask].tolist()