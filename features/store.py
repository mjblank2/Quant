"""
Feature Store - Centralized storage and retrieval of features with training/serving consistency
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, date, timedelta
from sqlalchemy import text
import logging
import hashlib
import pickle

from .registry import FeatureRegistry, registry

log = logging.getLogger(__name__)

class FeatureStore:
    """
    Centralized feature store ensuring consistency between training and live trading.
    Prevents training-serving skew by using identical feature computation logic.
    """
    
    def __init__(self, engine, feature_registry: Optional[FeatureRegistry] = None):
        self.engine = engine
        self.registry = feature_registry or registry
        self._feature_cache: Dict[str, Any] = {}
        
    def compute_features(self, symbols: List[str], start_date: Optional[date] = None, 
                        end_date: Optional[date] = None, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute features for given symbols and date range using registered feature definitions.
        Ensures identical computation logic for training and serving.
        """
        if not symbols:
            return pd.DataFrame()
        
        # Get all features if none specified
        if feature_names is None:
            feature_names = list(self.registry._features.keys())
        
        # Validate features exist
        missing_features = [name for name in feature_names if not self.registry.get_feature(name)]
        if missing_features:
            raise ValueError(f"Unknown features: {missing_features}")
        
        # Load base data
        base_data = self._load_base_data(symbols, start_date, end_date)
        if base_data.empty:
            return pd.DataFrame()
        
        # Compute features in dependency order
        feature_results = []
        
        for symbol, symbol_data in base_data.groupby('symbol'):
            symbol_data = symbol_data.sort_values('ts').copy()
            symbol_features = self._compute_symbol_features(symbol_data, feature_names)
            feature_results.append(symbol_features)
        
        return pd.concat(feature_results, ignore_index=True) if feature_results else pd.DataFrame()
    
    def get_latest_features(self, symbols: List[str], feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get the most recent features for given symbols"""
        if not symbols:
            return pd.DataFrame()
        
        # For live trading, we want the most recent features
        latest_date = self._get_latest_date()
        if latest_date is None:
            return pd.DataFrame()
        
        return self.compute_features(symbols, start_date=latest_date, end_date=latest_date, feature_names=feature_names)
    
    def store_features(self, features_df: pd.DataFrame) -> None:
        """Store computed features to database"""
        if features_df.empty:
            return
        
        from db import upsert_dataframe, Feature
        upsert_dataframe(features_df, Feature, ['symbol', 'ts'])
        log.info(f"Stored {len(features_df)} feature records")
    
    def get_feature_metadata(self, feature_name: str) -> Dict[str, Any]:
        """Get metadata about a feature"""
        feature_def = self.registry.get_feature(feature_name)
        if not feature_def:
            return {}
        
        return {
            'name': feature_def.name,
            'description': feature_def.description,
            'type': feature_def.feature_type,
            'dependencies': feature_def.dependencies,
            'lookback_days': feature_def.lookback_days,
            'update_frequency': feature_def.update_frequency,
            'data_sources': feature_def.data_sources,
            'version': feature_def.version,
            'tags': feature_def.tags
        }
    
    def validate_feature_consistency(self, training_features: pd.DataFrame, 
                                   serving_features: pd.DataFrame, tolerance: float = 1e-6) -> Dict[str, Any]:
        """Validate that features computed for training and serving are consistent"""
        validation_results = {
            'consistent': True,
            'errors': [],
            'warnings': []
        }
        
        # Check column consistency
        training_cols = set(training_features.columns)
        serving_cols = set(serving_features.columns)
        
        if training_cols != serving_cols:
            validation_results['consistent'] = False
            validation_results['errors'].append(f"Column mismatch: training={training_cols}, serving={serving_cols}")
        
        # Check value consistency for overlapping data
        common_cols = training_cols.intersection(serving_cols)
        numeric_cols = [col for col in common_cols if col not in ['symbol', 'ts']]
        
        if not training_features.empty and not serving_features.empty:
            # Find overlapping records
            training_key = training_features[['symbol', 'ts']].copy()
            serving_key = serving_features[['symbol', 'ts']].copy()
            training_key['key'] = training_key['symbol'] + '_' + training_key['ts'].astype(str)
            serving_key['key'] = serving_key['symbol'] + '_' + serving_key['ts'].astype(str)
            
            overlap_keys = set(training_key['key']).intersection(set(serving_key['key']))
            
            for key in list(overlap_keys)[:100]:  # Check up to 100 overlapping records
                train_row = training_features[training_key['key'] == key].iloc[0]
                serve_row = serving_features[serving_key['key'] == key].iloc[0]
                
                for col in numeric_cols:
                    if col in train_row and col in serve_row:
                        train_val = train_row[col]
                        serve_val = serve_row[col]
                        
                        if pd.isna(train_val) and pd.isna(serve_val):
                            continue
                        elif pd.isna(train_val) or pd.isna(serve_val):
                            validation_results['warnings'].append(f"NA mismatch in {col} for {key}")
                        elif abs(train_val - serve_val) > tolerance:
                            validation_results['consistent'] = False
                            validation_results['errors'].append(f"Value mismatch in {col} for {key}: {train_val} vs {serve_val}")
        
        return validation_results
    
    def _load_base_data(self, symbols: List[str], start_date: Optional[date], end_date: Optional[date]) -> pd.DataFrame:
        """Load base pricing and fundamental data needed for feature computation"""
        try:
            # Build date filter and parameters
            date_filter = ""
            params = {}
            
            if start_date:
                date_filter += " AND ts >= :start_date"
                params['start_date'] = start_date
            
            if end_date:
                date_filter += " AND ts <= :end_date"
                params['end_date'] = end_date
            
            # Create IN clause for symbols
            symbol_placeholders = ', '.join([f':sym_{i}' for i in range(len(symbols))])
            for i, symbol in enumerate(symbols):
                params[f'sym_{i}'] = symbol
            
            # Try loading pricing data with adj_close first
            price_sql_with_adj = f"""
                SELECT symbol, ts, open, close, COALESCE(adj_close, close) as adj_close, volume,
                       COALESCE(adj_close, close) AS price_feat
                FROM daily_bars 
                WHERE symbol IN ({symbol_placeholders}) {date_filter}
                ORDER BY symbol, ts
            """
            
            # Fallback SQL for databases without adj_close column
            price_sql_fallback = f"""
                SELECT symbol, ts, open, close, close as adj_close, volume,
                       close AS price_feat
                FROM daily_bars 
                WHERE symbol IN ({symbol_placeholders}) {date_filter}
                ORDER BY symbol, ts
            """
            
            try:
                prices = pd.read_sql_query(text(price_sql_with_adj), self.engine, params=params, parse_dates=['ts'])
            except Exception as e:
                if "adj_close" in str(e) and ("no such column" in str(e) or "does not exist" in str(e)):
                    log.warning("adj_close column not found in daily_bars table, using close price instead")
                    prices = pd.read_sql_query(text(price_sql_fallback), self.engine, params=params, parse_dates=['ts'])
                else:
                    raise
            
            if prices.empty:
                return pd.DataFrame()
            
            # Load shares outstanding
            shares_sql = f"""
                SELECT symbol, as_of, shares
                FROM shares_outstanding 
                WHERE symbol IN ({symbol_placeholders})
                ORDER BY symbol, as_of
            """
            
            shares = pd.read_sql_query(text(shares_sql), self.engine, params=params, parse_dates=['as_of'])
            
            # Merge shares with point-in-time logic
            if not shares.empty:
                shares_renamed = shares.rename(columns={'as_of': 'ts_shares'})
                prices = pd.merge_asof(
                    prices.sort_values(['symbol', 'ts']),
                    shares_renamed.sort_values(['symbol', 'ts_shares']),
                    left_on='ts', right_on='ts_shares', by='symbol', direction='backward'
                )
                prices['shares_out'] = prices['shares']
            else:
                prices['shares_out'] = np.nan
            
            return prices
            
        except Exception as e:
            log.error(f"Error loading base data: {e}")
            return pd.DataFrame()
    
    def _compute_symbol_features(self, symbol_data: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """Compute features for a single symbol"""
        result_data = symbol_data[['symbol', 'ts']].copy()
        
        # Add computed features
        for feature_name in feature_names:
            feature_def = self.registry.get_feature(feature_name)
            if not feature_def:
                continue
            
            try:
                # Ensure we have enough data for lookback
                if len(symbol_data) < feature_def.lookback_days:
                    result_data[feature_name] = np.nan
                    continue
                
                # Compute feature
                feature_values = feature_def.computation(symbol_data)
                
                # Handle series/scalar results
                if isinstance(feature_values, pd.Series):
                    result_data[feature_name] = feature_values.values
                else:
                    result_data[feature_name] = feature_values
                    
            except Exception as e:
                log.warning(f"Error computing {feature_name} for {symbol_data['symbol'].iloc[0]}: {e}")
                result_data[feature_name] = np.nan
        
        return result_data
    
    def _get_latest_date(self) -> Optional[date]:
        """Get the most recent date with pricing data"""
        try:
            result = pd.read_sql_query(text("SELECT MAX(ts) as max_date FROM daily_bars"), self.engine)
            return result['max_date'].iloc[0] if not result.empty else None
        except Exception:
            return None