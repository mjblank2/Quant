"""
Cross-validation utilities for time-series data.

This module implements purged k-fold cross-validation with an embargo as well
as combinatorial purged cross-validation (CPCV), both designed to avoid
look-ahead bias when evaluating models on financial time-series. The
implementations are based on the methods described in Marcos López de Prado's
*Advances in Financial Machine Learning* and related research. The goal is
to ensure that observations used for training and testing do not overlap
within the label formation window and that a buffer (embargo) is applied
following test periods to prevent information leakage.

Usage example::

    from validation.cv import PurgedKFoldEmbargo, CombinatorialPurgedCV

    cv = PurgedKFoldEmbargo(n_splits=5, label_horizon=5, embargo_pct=0.05)
    for train_idx, test_idx in cv.split(features, timestamps):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])

    cpcv = CombinatorialPurgedCV(n_splits=6, k_test=2, label_horizon=5, embargo_pct=0.05)
    for train_idx, test_idx in cpcv.split(features, timestamps):
        ...

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd

@dataclass
class PurgedKFoldEmbargo:
    """Time-series aware k-fold cross-validation with purging and embargo.

    Parameters
    ----------
    n_splits : int
        Number of splits/folds.
    label_horizon : int
        Number of observations in the label formation window. Samples whose
        indices fall within ``label_horizon`` of the test window are purged
        from the training set.
    embargo_pct : float
        Fraction of the dataset length to embargo after each test fold. The
        embargo prevents training samples that are close in time from
        overlapping with test samples that might leak information.

    Notes
    -----
    The data must be ordered by time. The ``split`` method accepts either
    a one-dimensional array of timestamps (``pd.Series``) or a two-dimensional
    feature matrix with a ``ts`` column. The returned indices can be used
    directly with ``numpy`` arrays or ``pandas`` DataFrame/Series.
    """

    n_splits: int = 5
    label_horizon: int = 1
    embargo_pct: float = 0.0

    def split(self, X: Optional[pd.DataFrame] = None, timestamps: Optional[pd.Series] = None
              ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for each fold.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Feature matrix with either a ``ts`` column or a datetime index.
        timestamps : pd.Series, optional
            If provided, a one-dimensional series of timestamps corresponding
            to rows in ``X``. When ``timestamps`` is given, ``X`` may be
            omitted.

        Yields
        ------
        (train_indices, test_indices) : Tuple[np.ndarray, np.ndarray]
            The indices for the training and test sets of each fold.
        """
        if timestamps is None:
            if X is None:
                raise ValueError("Either X or timestamps must be provided.")
            if 'ts' in X.columns:
                timestamps = pd.to_datetime(X['ts'])
            elif isinstance(X.index, pd.DatetimeIndex):
                timestamps = X.index.to_series()
            else:
                raise ValueError("X must have a 'ts' column or datetime index if timestamps is not provided.")
        else:
            timestamps = pd.to_datetime(timestamps)

        n = len(timestamps)
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        fold_sizes = (n // self.n_splits) * np.ones(self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        indices = np.arange(n)
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            # Purging: remove samples within label_horizon of test set from training
            purge_start = max(0, start - self.label_horizon)
            purge_end = min(n, stop + self.label_horizon)
            train_mask = np.ones(n, dtype=bool)
            train_mask[purge_start:purge_end] = False
            # Embargo: remove a fraction of the remaining samples after test set
            embargo = int(np.ceil(n * self.embargo_pct))
            if embargo > 0:
                embargo_start = stop
                embargo_end = min(n, stop + embargo)
                train_mask[embargo_start:embargo_end] = False
            train_idx = indices[train_mask]
            yield (train_idx, test_idx)
            current = stop


@dataclass
class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (CPCV).

    Generates all combinations of ``k_test`` test folds out of ``n_splits``
    folds, purges overlapping training samples within the ``label_horizon``
    of test windows, and applies an optional embargo percentage. CPCV
    provides a distribution of out-of-sample metrics by evaluating many
    different contiguous test windows.

    Parameters
    ----------
    n_splits : int
        Total number of folds in which to partition the data.
    k_test : int
        Number of folds to use for testing in each combination. ``k_test``
        must be less than ``n_splits``.
    label_horizon : int
        Size of the label formation window used for purging.
    embargo_pct : float
        Fraction of the dataset length to embargo after each test block.
    """

    n_splits: int = 6
    k_test: int = 2
    label_horizon: int = 1
    embargo_pct: float = 0.0

    def split(self, X: Optional[pd.DataFrame] = None, timestamps: Optional[pd.Series] = None
              ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield train/test indices for each combination of test folds.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Feature matrix with a ``ts`` column or datetime index.
        timestamps : pd.Series, optional
            One-dimensional series of timestamps corresponding to rows in ``X``.

        Yields
        ------
        (train_indices, test_indices) : Tuple[np.ndarray, np.ndarray]
            Indices of training and test samples for each CPCV split.
        """
        if timestamps is None:
            if X is None:
                raise ValueError("Either X or timestamps must be provided.")
            if 'ts' in X.columns:
                timestamps = pd.to_datetime(X['ts'])
            elif isinstance(X.index, pd.DatetimeIndex):
                timestamps = X.index.to_series()
            else:
                raise ValueError("X must have a 'ts' column or datetime index if timestamps is not provided.")
        else:
            timestamps = pd.to_datetime(timestamps)

        n = len(timestamps)
        if self.n_splits < 2 or self.k_test < 1 or self.k_test >= self.n_splits:
            raise ValueError("Invalid n_splits/k_test settings.")
        fold_sizes = (n // self.n_splits) * np.ones(self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        indices = np.arange(n)
        # Partition indices into folds
        folds: List[np.ndarray] = []
        current = 0
        for size in fold_sizes:
            folds.append(indices[current:current + size])
            current += size
        # Iterate over all combinations of test folds
        import itertools
        for combo in itertools.combinations(range(self.n_splits), self.k_test):
            test_idx = np.concatenate([folds[i] for i in combo])
            test_start = min(idx for idx in test_idx)
            test_end = max(idx for idx in test_idx)
            # Purge: remove samples within label_horizon of any test index
            train_mask = np.ones(n, dtype=bool)
            for idx in test_idx:
                start = max(0, idx - self.label_horizon)
                end = min(n, idx + self.label_horizon + 1)
                train_mask[start:end] = False
            # Embargo: apply after the test block
            embargo = int(np.ceil(n * self.embargo_pct))
            if embargo > 0:
                emb_start = test_end + 1
                emb_end = min(n, test_end + 1 + embargo)
                train_mask[emb_start:emb_end] = False
            train_idx = indices[train_mask]
            yield (train_idx, test_idx)