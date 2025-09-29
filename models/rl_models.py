"""
Reinforcement learning models for quantitative trading.

This module defines lightweight reinforcement learning style models that fit
into the scikit‑learn API.  Given the limited package availability in this
environment (no gym, stable_baselines or deep learning frameworks), the
implementation here provides a simple Q‑table approach.  It discretizes
training returns into bins and learns an expected reward for each action
(long, hold, short) conditioned on a discrete state.  At prediction time,
states are inferred from the first feature (e.g. recent return) and the
action with the highest expected reward determines the predicted signal.

While this implementation is greatly simplified compared to full RL
algorithms, it demonstrates the structure needed to incorporate
reinforcement‑learning inspired models into the existing pipeline.  You can
extend it with more sophisticated state representations or reward
functions as needed.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any


class QTableRegressor:
    """A simple Q‑table regressor compatible with scikit‑learn pipelines.

    This model discretizes the target variable into a fixed number of bins
    and learns expected rewards for three discrete actions: long, hold, and
    short.  The model does not learn from features directly; instead, it
    infers the state during prediction from the first feature column
    (assumed to be a recent return).  The action with the highest
    expected reward determines the sign of the prediction.

    Parameters
    ----------
    n_bins : int, optional (default=10)
        Number of quantile bins used to discretize the target variable.
    random_state : int or None (default=None)
        Random seed for reproducibility.  Currently unused but included to
        adhere to scikit‑learn API conventions.

    Notes
    -----
    The class implements ``fit`` and ``predict`` methods to be used in a
    ``Pipeline``.  No hyperparameters beyond ``n_bins`` and ``random_state``
    are exposed.  The ``sample_weight`` argument is accepted in ``fit``
    for API compatibility but is not used in this simplified implementation.
    """

    def __init__(self, n_bins: int = 10, random_state: Optional[int] = None) -> None:
        self.n_bins = n_bins
        self.random_state = random_state
        self.bins_: Optional[np.ndarray] = None
        self.q_table_: Optional[np.ndarray] = None

    # scikit‑learn API compliance
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"n_bins": self.n_bins, "random_state": self.random_state}

    def set_params(self, **params: Any) -> 'QTableRegressor':
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _discretize(self, y: np.ndarray) -> np.ndarray:
        """Discretize continuous targets into quantile bins.

        Parameters
        ----------
        y : array‑like
            Continuous target values.

        Returns
        -------
        states : ndarray of ints
            Discrete state indices for each sample.
        """
        # Compute quantile bin edges.  Duplicate values are handled by
        # ``numpy.quantile``, which will still return monotonically
        # increasing edges for large enough ``n_bins``.
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        bins = np.quantile(y, quantiles)
        # Ensure unique bin edges by adding a tiny jitter if necessary.
        # This prevents ``digitize`` from assigning all points to the
        # highest bin when duplicates are present.
        eps = 1e‑8
        for i in range(1, len(bins)):
            if bins[i] <= bins[i - 1]:
                bins[i] = bins[i - 1] + eps
        self.bins_ = bins
        # Digitize returns indices from 1 to n_bins; subtract 1 to index 0..n_bins‑1
        states = np.digitize(y, bins[1:-1], right=False)
        return states

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'QTableRegressor':
        """Fit the Q‑table based on training targets.

        Parameters
        ----------
        X : array‑like, shape (n_samples, n_features)
            Training features.  Unused in this implementation.
        y : array‑like, shape (n_samples,)
            Training target values (e.g. forward returns).
        sample_weight : array‑like, optional
            Sample weights.  Ignored in this simplified implementation.

        Returns
        -------
        self : QTableRegressor
            Fitted estimator.
        """
        y = np.asarray(y).astype(float)
        # Discretize targets into states
        states = self._discretize(y)
        # Initialize Q‑table: shape (n_bins, 3) for actions [long, hold, short]
        q_table = np.zeros((self.n_bins, 3), dtype=float)
        # Compute expected rewards per state and action
        for s in range(self.n_bins):
            idx = np.where(states == s)[0]
            if idx.size == 0:
                continue
            ys = y[idx]
            # Expected reward when going long: positive returns only
            pos = ys[ys > 0.0]
            long_reward = pos.mean() if pos.size > 0 else 0.0
            # Expected reward when holding: zero by definition (no trade)
            hold_reward = 0.0
            # Expected reward when going short: negative of negative returns
            neg = ys[ys < 0.0]
            short_reward = (‑neg).mean() if neg.size > 0 else 0.0
            q_table[s, 0] = long_reward
            q_table[s, 1] = hold_reward
            q_table[s, 2] = short_reward
        self.q_table_ = q_table
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict signals based on the fitted Q‑table.

        The prediction is the sign of the action with the highest expected
        reward for the discretized state inferred from the first feature.

        Parameters
        ----------
        X : array‑like, shape (n_samples, n_features)
            Feature matrix for which to generate predictions.  Only the
            first column is used to infer the state.

        Returns
        -------
        signals : ndarray, shape (n_samples,)
            Signals taking values {‑1, 0, +1}, where +1 indicates a
            recommendation to go long, ‑1 short, and 0 hold.
        """
        if self.bins_ is None or self.q_table_ is None:
            raise ValueError("QTableRegressor must be fitted before calling predict().")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(‑1, 1)
        # Use first feature (e.g. recent return) to infer state
        x_vals = X[:, 0]
        # Assign states based on the fitted bins; clip to valid range
        states = np.digitize(x_vals, self.bins_[1:-1], right=False)
        # Map states to actions and then to signal values
        signals = np.zeros(len(states), dtype=float)
        for i, s in enumerate(states):
            s_idx = max(0, min(int(s), self.n_bins ‑ 1))
            # action 0=long, 1=hold, 2=short
            a = int(np.argmax(self.q_table_[s_idx]))
            if a == 0:
                signals[i] = 1.0
            elif a == 2:
                signals[i] = ‑1.0
            else:
                signals[i] = 0.0
        return signals