
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CrossSectionalNormalizer(BaseEstimator, TransformerMixin):
    """Winsorize + standardize features cross-sectionally. Stateless."""
    def __init__(self, winsorize_tails: float = 0.05):
        self.winsorize_tails = winsorize_tails

    def fit(self, X, y=None):
        return self

    def _quantile(self, X, q):
        try:
            return np.nanquantile(X, q, axis=0, method='linear')
        except TypeError:
            return np.nanquantile(X, q, axis=0, interpolation='linear')

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.winsorize_tails > 0:
            lower = self._quantile(X, self.winsorize_tails)
            upper = self._quantile(X, 1.0 - self.winsorize_tails)
            X = np.clip(X, lower, upper)
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std = np.where(std == 0, 1.0, std)
        X = (X - mean) / std
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
