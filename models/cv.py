from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Iterable, List, Tuple, Sequence, Union

ArrayLike = Union[pd.Series, pd.Index, Sequence[pd.Timestamp], Sequence[pd.datetime], Sequence]

def _unique_sorted_dates(dates: ArrayLike) -> pd.DatetimeIndex:
    s = pd.to_datetime(pd.Index(dates))
    return pd.DatetimeIndex(sorted(s.unique()))

def combinatorial_purged_cv(
    dates: ArrayLike,
    n_groups: int = 6,
    k_test_groups: int = 2,
    embargo: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """True Combinatorial Purged Cross-Validation (CPCV).

    Parameters
    ----------
    dates : array-like
        Per-sample timestamps (length = n_samples). Duplicates are allowed.
    n_groups : int, default 6
        Number of contiguous groups the ordered *unique* dates are split into.
    k_test_groups : int, default 2
        Number of groups used jointly for each test fold (combinatorial choose).
    embargo : int, default 5
        Number of *unique-date* indices to embargo before and after the test dates
        from the training set (purging temporal leakage).

    Returns
    -------
    folds : list of (train_idx, test_idx)
        Each contains numpy arrays of *sample indices* for train and test.

    Notes
    -----
    - This splitter operates at the unique-date level (not raw row indices).
    - It first partitions the ordered unique dates into ``n_groups`` contiguous
      slices, then iterates over all ``combinations`` of size ``k_test_groups``.
    - For each fold, all samples whose date falls into the chosen test groups are
      in the test set. Training samples are those whose date indices are not in
      the test groups **and** not within the embargo window around any test date.
    - If your labels have longer information overlap than a single day, increase
      ``embargo`` accordingly.
    """
    # Map each sample to a unique-date position
    u = _unique_sorted_dates(dates)
    m = len(u)
    if m == 0:
        return []

    # Partition unique-date positions into contiguous groups
    pos = np.arange(m)
    groups = np.array_split(pos, n_groups)

    # Map each unique date to its position
    pos_map = {d: i for i, d in enumerate(u)}

    # Sample-level position for quick masking
    dates_idx = pd.to_datetime(pd.Index(dates))
    sample_pos = dates_idx.map(pos_map).to_numpy()

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for comb in combinations(range(n_groups), k_test_groups):
        test_pos = np.concatenate([groups[g] for g in comb]) if k_test_groups > 1 else groups[comb[0]]
        test_pos = np.unique(test_pos)

        # Build embargoed training mask on the unique-date axis
        train_ok_pos = np.ones(m, dtype=bool)

        # Purge an embargo window around *every* test position
        for p in test_pos:
            lo = max(0, p - embargo)
            hi = min(m, p + embargo + 1)
            train_ok_pos[lo:hi] = False

        # Sample-level masks
        is_test_sample = np.isin(sample_pos, test_pos)
        is_train_sample = ~is_test_sample & train_ok_pos[sample_pos]

        train_idx = np.where(is_train_sample)[0]
        test_idx = np.where(is_test_sample)[0]

        if len(test_idx) == 0 or len(train_idx) == 0:
            # Skip degenerate folds (can happen with very short histories)
            continue

        folds.append((train_idx, test_idx))

    return folds
