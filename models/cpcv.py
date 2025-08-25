from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple

def combinatorial_purged_cv_dates(dates: List[pd.Timestamp], n_groups: int = 6, embargo: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return index pairs (train_idx, test_idx) for CPCV over ordered unique dates."""
    u = sorted(pd.to_datetime(pd.Series(dates).unique()))
    m = len(u)
    g = np.array_split(np.arange(m), n_groups)
    folds = []
    for i in range(n_groups):
        test_idx = g[i]
        # embargo around test dates
        lo = max(test_idx[0]-embargo, 0)
        hi = min(test_idx[-1]+embargo+1, m)
        train_mask = np.ones(m, dtype=bool)
        train_mask[lo:hi] = False
        train_idx = np.where(train_mask)[0]
        folds.append((train_idx, test_idx))
    return folds
