from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted_ = False
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self.is_fitted_ = True
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted before transform")
        return X

class CrossSectionalRankNormalizer(FeatureTransformer):
    """Cross-sectional normalization for DataFrames (no sklearn Pipelines).


    method: 'rank' | 'zscore' | 'robust'

    Avoids name collision with models.transformers.CrossSectionalNormalizer.
    """
    def __init__(self, method: str = 'rank', clip_std: float = 3.0):
        super().__init__()
        self.method = method
        self.clip_std = float(clip_std)
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if 'ts' not in X.columns:
            raise ValueError("CrossSectionalRankNormalizer requires a 'ts' column")
        self.feature_columns_ = [c for c in X.columns if c not in ['symbol','ts']]
        self.is_fitted_ = True
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        super().transform(X)
        out = X.copy()
        for ts, g in X.groupby('ts'):
            for col in self.feature_columns_:
                if col not in g.columns: 
                    continue
                v = g[col].astype(float)
                if self.method == 'rank':
                    normed = v.rank(pct=True) - 0.5
                elif self.method == 'zscore':
                    normed = (v - v.mean()) / (v.std(ddof=0) + 1e-8)
                    normed = np.clip(normed, -self.clip_std, self.clip_std)
                elif self.method == 'robust':
                    med = v.median()
                    mad = (v - med).abs().median()
                    normed = (v - med) / (mad + 1e-8)
                    normed = np.clip(normed, -self.clip_std, self.clip_std)
                else:
                    raise ValueError(f"Unknown method: {self.method}")
                out.loc[g.index, col] = normed.reindex(g.index)
        return out

class TimeSeriesLagFeatures(FeatureTransformer):
    def __init__(self, lags: List[int], columns: Optional[List[str]] = None):
        super().__init__()
        self.lags = list(lags)
        self.columns = columns
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if 'symbol' not in X.columns:
            raise ValueError("TimeSeriesLagFeatures requires 'symbol' column")
        self.feature_columns_ = self.columns or [c for c in X.columns if c not in ['symbol','ts']]
        self.is_fitted_ = True
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        super().transform(X)
        out = X.copy()
        for col in self.feature_columns_:
            if col not in X.columns: 
                continue
            for lag in self.lags:
                out[f"{col}_lag{lag}"] = X.groupby('symbol')[col].shift(lag)
        return out

class TechnicalIndicators(FeatureTransformer):
    def __init__(self, price_col: str = 'price_feat', volume_col: str = 'volume'):
        super().__init__()
        self.price_col = price_col
        self.volume_col = volume_col
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        need = [self.price_col]
        if self.volume_col in X.columns:
            need.append(self.volume_col)
        mis = [c for c in need if c not in X.columns]
        if mis:
            raise ValueError(f"Missing required columns: {mis}")
        self.is_fitted_ = True
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        super().transform(X)
        out = X.copy()
        for sym, g in X.groupby('symbol'):
            g = g.sort_values('ts')
            p = g[self.price_col].astype(float)
            out.loc[g.index, 'rsi_14'] = _rsi(p, 14)
            sma20 = p.rolling(20).mean()
            std20 = p.rolling(20).std()
            out.loc[g.index, 'sma_20'] = sma20
            out.loc[g.index, 'sma_50'] = p.rolling(50).mean()
            out.loc[g.index, 'bb_upper'] = sma20 + 2*std20
            out.loc[g.index, 'bb_lower'] = sma20 - 2*std20
            den = (out.loc[g.index, 'bb_upper'] - out.loc[g.index, 'bb_lower']).replace(0, np.nan)
            out.loc[g.index, 'bb_position'] = (p - out.loc[g.index, 'bb_lower']) / den
            ema12 = p.ewm(span=12).mean()
            ema26 = p.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            out.loc[g.index, 'macd'] = macd
            out.loc[g.index, 'macd_signal'] = signal
            out.loc[g.index, 'macd_histogram'] = macd - signal
            if self.volume_col in g.columns:
                v = g[self.volume_col].astype(float)
                out.loc[g.index, 'volume_sma_20'] = v.rolling(20).mean()
                out.loc[g.index, 'volume_ratio'] = v / out.loc[g.index, 'volume_sma_20']
        return out

def _rsi(p: pd.Series, window: int) -> pd.Series:
    d = p.diff()
    gain = d.where(d > 0, 0.0)
    loss = -d.where(d < 0, 0.0)
    ag = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    al = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = ag / (al + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs))

class AlternativeDataFeatures(FeatureTransformer):
    def __init__(self, sentiment_decay: float = 0.9, news_window: int = 5, random_state: Optional[int] = 42):
        super().__init__()
        self.sentiment_decay = float(sentiment_decay)
        self.news_window = int(news_window)
        self.random_state = random_state
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self.rng_ = np.random.default_rng(self.random_state)
        self.is_fitted_ = True
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        super().transform(X)
        out = X.copy()
        n = len(out)
        out['news_sentiment'] = self.rng_.normal(0, 0.1, n)
        out['social_sentiment'] = self.rng_.normal(0, 0.15, n)
        out['esg_score'] = self.rng_.uniform(0, 100, n)
        out['supply_chain_risk'] = self.rng_.uniform(0, 1, n)
        return out
