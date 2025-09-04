from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CrossSectionalNormalizer(BaseEstimator, TransformerMixin):
    """Cross-sectional winsorization + standardization (sklearn-compatible).

    - Stateless across fits to avoid lookahead in cross-sectional usage.
    - Works with numpy arrays; callers using pandas should pass ``df.values``.
    """
    def __init__(self, winsorize_tails: float = 0.05):
        self.winsorize_tails = float(winsorize_tails)

    def fit(self, X, y=None):
        return self

    def _quantile(self, X, q):
        try:
            return np.nanquantile(X, q, axis=0, method='linear')
        except TypeError:
            return np.nanquantile(X, q, axis=0, interpolation='linear')

    def transform(self, X):
        import numpy as _np  # local import to ensure numpy exists where used
        if hasattr(X, "values"):
            X = X.values
        X = _np.asarray(X, dtype=float)
        if self.winsorize_tails > 0:
            lower = self._quantile(X, self.winsorize_tails)
            upper = self._quantile(X, 1.0 - self.winsorize_tails)
            X = _np.clip(X, lower, upper)
        mean = _np.nanmean(X, axis=0)
        std = _np.nanstd(X, axis=0)
        std = _np.where(std == 0, 1.0, std)
        X = (X - mean) / std
        return _np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
