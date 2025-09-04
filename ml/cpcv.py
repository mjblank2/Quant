
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

@dataclass(frozen=True)
class CPCVFold:
    train_idx: np.ndarray
    test_idx: np.ndarray

def _group_by_unique_dates(dates: Sequence[pd.Timestamp]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Map sample rows -> unique-date index, and build an index list per unique date.

    Returns:
        unique_dates_index: array of shape (n_samples,) with integer index of the unique date.
        rows_per_udate: list where rows_per_udate[i] = np.array of row indices belonging to unique date i.
    """
    dser = pd.to_datetime(pd.Series(dates), utc=False).reset_index(drop=True)
    udates = dser.drop_duplicates().reset_index(drop=True)
    u_lookup = {d: i for i, d in enumerate(udates)}
    unique_dates_index = dser.map(u_lookup).values
    rows_per_udate: List[List[int]] = [[] for _ in range(len(udates))]
    for row_i, ud_i in enumerate(unique_dates_index):
        rows_per_udate[int(ud_i)].append(row_i)
    rows_per_udate = [np.array(v, dtype=int) for v in rows_per_udate]
    return unique_dates_index, rows_per_udate

def combinatorial_purged_cv(
    dates: Sequence[pd.Timestamp],
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    True Combinatorial Purged Cross-Validation (López de Prado).
    - Split ordered unique dates into `n_groups` contiguous blocks.
    - For each combination of `n_test_groups` blocks as test, use the rest as train,
      and apply an embargo of `embargo` unique-date steps around every test block
      (removed from train). Returns sample-level indices for sklearn compatibility.

    Args:
        dates: per-row timestamps (length = n_samples).
        n_groups: number of contiguous groups to form across unique sorted dates.
        n_test_groups: number of groups to use as test in each fold.
        embargo: number of unique-date steps to exclude from the train set around each test block.

    Returns:
        List of (train_idx, test_idx) np.ndarrays over original row indices.
    """
    if n_test_groups <= 0 or n_test_groups >= n_groups:
        raise ValueError("n_test_groups must be in [1, n_groups-1]")

    # Map rows to unique dates and build per-date row indices
    _, rows_per_udate = _group_by_unique_dates(dates)
    m = len(rows_per_udate)  # number of unique dates
    if m == 0:
        return []

    # Build contiguous groups of unique-date indices
    u_idx = np.arange(m)
    groups = np.array_split(u_idx, n_groups)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    for combo in itertools.combinations(range(n_groups), n_test_groups):
        # Test unique-date indices are the union of the selected blocks
        test_u = np.concatenate([groups[i] for i in combo])
        test_u = np.unique(test_u)

        # Build a mask of trainable unique dates (start with all True)
        train_mask = np.ones(m, dtype=bool)
        # Remove test dates
        train_mask[test_u] = False

        # Purge/embargo: for each contiguous run in test_u, zero-out a window around it
        if embargo > 0 and len(test_u) > 0:
            # Identify contiguous segments within test_u
            diffs = np.diff(test_u)
            breaks = np.where(diffs != 1)[0]
            segment_starts = np.r_[0, breaks + 1]
            segment_ends = np.r_[breaks, len(test_u) - 1]
            for s, e in zip(segment_starts, segment_ends):
                lo = max(int(test_u[s]) - embargo, 0)
                hi = min(int(test_u[e]) + embargo, m - 1)
                train_mask[lo:hi+1] = False

        # Map unique-date indices back to row indices
        test_rows = np.concatenate([rows_per_udate[i] for i in test_u]) if len(test_u) else np.array([], dtype=int)
        train_u = np.where(train_mask)[0]
        train_rows = np.concatenate([rows_per_udate[i] for i in train_u]) if len(train_u) else np.array([], dtype=int)

        # Sort for reproducibility
        test_rows = np.sort(test_rows)
        train_rows = np.sort(train_rows)

        folds.append((train_rows, test_rows))

    return folds

class CombinatorialPurgedCV(BaseCrossValidator):
    """Sklearn-compatible splitter implementing true CPCV with embargo."""
    def __init__(self, n_groups: int = 6, n_test_groups: int = 2, embargo: int = 5, dates: Sequence[pd.Timestamp] | None = None):
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo = embargo
        self.dates = dates  # if not provided in split()

    def get_n_splits(self, X=None, y=None, groups=None):
        # Depends on number of combinations; compute lazily in split().
        if self.dates is None and X is None:
            return 0
        d = self.dates if self.dates is not None else getattr(X, 'ts', None)
        if d is None:
            return 0
        d = pd.to_datetime(pd.Series(d))
        m = d.drop_duplicates().shape[0]
        import math
        from math import comb
        return comb(self.n_groups, self.n_test_groups)

    def split(self, X, y=None, groups=None):
        if self.dates is not None:
            dates = self.dates
        else:
            if hasattr(X, 'ts'):
                dates = X['ts']
            else:
                raise ValueError("Provide 'dates' at init or ensure X has a 'ts' column with timestamps.")
        folds = combinatorial_purged_cv(dates, n_groups=self.n_groups, n_test_groups=self.n_test_groups, embargo=self.embargo)
        for tr, te in folds:
            yield tr, te
