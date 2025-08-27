"""
Feature Transformations - Reusable feature engineering components
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Union, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

log = logging.getLogger(__name__)

class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Base class for feature transformations with fit/transform pattern"""
    
    def __init__(self):
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the transformer"""
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the features"""
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted before transform")
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)

class CrossSectionalNormalizer(FeatureTransformer):
    """
    Cross-sectional normalization: rank-based transformation within each time period.
    Ensures features are comparable across different market regimes.
    """
    
    def __init__(self, method: str = 'rank', clip_std: float = 3.0):
        super().__init__()
        self.method = method  # 'rank', 'zscore', 'robust'
        self.clip_std = clip_std
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the normalizer (no parameters to learn for cross-sectional)"""
        if 'ts' not in X.columns:
            raise ValueError("CrossSectionalNormalizer requires 'ts' column")
        
        self.feature_columns_ = [col for col in X.columns if col not in ['symbol', 'ts']]
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply cross-sectional normalization"""
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted before transform")
        
        result = X.copy()
        
        for ts, group in X.groupby('ts'):
            for col in self.feature_columns_:
                if col not in group.columns:
                    continue
                
                values = group[col].dropna()
                if len(values) < 2:
                    continue
                
                if self.method == 'rank':
                    # Rank-based normalization (percentile)
                    normalized = values.rank(pct=True) - 0.5  # Center around 0
                elif self.method == 'zscore':
                    # Z-score normalization
                    normalized = (values - values.mean()) / (values.std() + 1e-8)
                    normalized = np.clip(normalized, -self.clip_std, self.clip_std)
                elif self.method == 'robust':
                    # Robust normalization using median and MAD
                    median = values.median()
                    mad = np.abs(values - median).median()
                    normalized = (values - median) / (mad + 1e-8)
                    normalized = np.clip(normalized, -self.clip_std, self.clip_std)
                else:
                    raise ValueError(f"Unknown normalization method: {self.method}")
                
                # Update the result dataframe
                result.loc[group.index, col] = normalized.reindex(group.index)
        
        return result

class TimeSeriesLagFeatures(FeatureTransformer):
    """Create lagged versions of features for time series analysis"""
    
    def __init__(self, lags: List[int], columns: Optional[List[str]] = None):
        super().__init__()
        self.lags = lags
        self.columns = columns
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the lag transformer"""
        if 'symbol' not in X.columns:
            raise ValueError("TimeSeriesLagFeatures requires 'symbol' column for grouping")
        
        self.feature_columns_ = self.columns or [col for col in X.columns if col not in ['symbol', 'ts']]
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted before transform")
        
        result = X.copy()
        
        for col in self.feature_columns_:
            if col not in X.columns:
                continue
            
            for lag in self.lags:
                lag_col = f"{col}_lag{lag}"
                result[lag_col] = X.groupby('symbol')[col].shift(lag)
        
        return result

class TechnicalIndicators(FeatureTransformer):
    """Compute common technical indicators"""
    
    def __init__(self, price_col: str = 'price_feat', volume_col: str = 'volume'):
        super().__init__()
        self.price_col = price_col
        self.volume_col = volume_col
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the technical indicators transformer"""
        required_cols = [self.price_col]
        if self.volume_col in X.columns:
            required_cols.append(self.volume_col)
        
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators"""
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted before transform")
        
        result = X.copy()
        
        for symbol, group in X.groupby('symbol'):
            group = group.sort_values('ts')
            prices = group[self.price_col]
            
            # RSI
            result.loc[group.index, 'rsi_14'] = self._compute_rsi(prices, 14)
            
            # Moving averages
            result.loc[group.index, 'sma_20'] = prices.rolling(20).mean()
            result.loc[group.index, 'sma_50'] = prices.rolling(50).mean()
            
            # Bollinger Bands
            sma_20 = prices.rolling(20).mean()
            std_20 = prices.rolling(20).std()
            result.loc[group.index, 'bb_upper'] = sma_20 + 2 * std_20
            result.loc[group.index, 'bb_lower'] = sma_20 - 2 * std_20
            result.loc[group.index, 'bb_position'] = (prices - result.loc[group.index, 'bb_lower']) / (result.loc[group.index, 'bb_upper'] - result.loc[group.index, 'bb_lower'])
            
            # MACD
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            result.loc[group.index, 'macd'] = macd
            result.loc[group.index, 'macd_signal'] = signal
            result.loc[group.index, 'macd_histogram'] = macd - signal
            
            # Volume indicators if available
            if self.volume_col in group.columns:
                volumes = group[self.volume_col]
                result.loc[group.index, 'volume_sma_20'] = volumes.rolling(20).mean()
                result.loc[group.index, 'volume_ratio'] = volumes / result.loc[group.index, 'volume_sma_20']
        
        return result
    
    def _compute_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Compute RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi

class AlternativeDataFeatures(FeatureTransformer):
    """Features from alternative data sources"""
    
    def __init__(self, sentiment_decay: float = 0.9, news_window: int = 5):
        super().__init__()
        self.sentiment_decay = sentiment_decay
        self.news_window = news_window
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit alternative data transformer"""
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform with alternative data features"""
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted before transform")
        
        result = X.copy()
        
        # Placeholder for alternative data features
        # In practice, these would integrate with external data sources
        
        # News sentiment (would come from NLP processing)
        result['news_sentiment'] = np.random.normal(0, 0.1, len(result))  # Placeholder
        
        # Social media sentiment
        result['social_sentiment'] = np.random.normal(0, 0.15, len(result))  # Placeholder
        
        # ESG scores (would come from ESG data providers)
        result['esg_score'] = np.random.uniform(0, 100, len(result))  # Placeholder
        
        # Supply chain disruption indicators
        result['supply_chain_risk'] = np.random.uniform(0, 1, len(result))  # Placeholder
        
        return result