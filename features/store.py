
from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from .registry import FeatureRegistry, registry
from utils.price_utils import select_price_as

log = logging.getLogger(__name__)

class FeatureStore:
    """Centralized feature store ensuring consistency between training and live trading."""
    def __init__(self, engine, feature_registry: Optional[FeatureRegistry] = None):
        self.engine = engine
        self.registry = feature_registry or registry
        self._feature_cache: Dict[str, Any] = {}

    def compute_features(self, symbols: List[str], start_date: Optional[date] = None,
                         end_date: Optional[date] = None, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        if feature_names is None:
            feature_names = list(self.registry._features.keys())

        missing = [n for n in feature_names if not self.registry.get_feature(n)]
        if missing:
            raise ValueError(f"Unknown features: {missing}")

        base_data = self._load_base_data(symbols, start_date, end_date)
        if base_data.empty:
            return pd.DataFrame()

        feature_results = []
        for symbol, symbol_data in base_data.groupby('symbol'):
            symbol_data = symbol_data.sort_values('ts').copy()
            symbol_features = self._compute_symbol_features(symbol_data, feature_names)
            feature_results.append(symbol_features)
        return pd.concat(feature_results, ignore_index=True) if feature_results else pd.DataFrame()

    def get_latest_features(self, symbols: List[str], feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        latest_date = self._get_latest_date()
        if latest_date is None:
            return pd.DataFrame()
        return self.compute_features(symbols, start_date=latest_date, end_date=latest_date, feature_names=feature_names)

    def store_features(self, features_df: pd.DataFrame) -> None:
        if features_df.empty:
            return
        from db import upsert_dataframe, Feature  # type: ignore
        features_df = (
            features_df.sort_values('ts')
            .drop_duplicates(['symbol', 'ts'], keep='last')
            .reset_index(drop=True)
        )
       upsert_dataframe(features_df, Feature, ['symbol','ts'])
        log.info(f"Stored {len(features_df)} feature records")

    def get_feature_metadata(self, feature_name: str) -> Dict[str, Any]:
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

    def validate_feature_consistency(self, training_features: pd.DataFrame, serving_features: pd.DataFrame, tolerance: float = 1e-6) -> Dict[str, Any]:
        validation_results = {'consistent': True, 'errors': [], 'warnings': []}
        training_cols = set(training_features.columns)
        serving_cols = set(serving_features.columns)
        if training_cols != serving_cols:
            validation_results['consistent'] = False
            validation_results['errors'].append(f"Column mismatch: training={training_cols}, serving={serving_cols}")
        common_cols = training_cols.intersection(serving_cols)
        numeric_cols = [col for col in common_cols if col not in ['symbol', 'ts']]
        if not training_features.empty and not serving_features.empty:
            training_key = training_features[['symbol', 'ts']].copy()
            serving_key = serving_features[['symbol', 'ts']].copy()
            training_key['key'] = training_key['symbol'] + '_' + training_key['ts'].astype(str)
            serving_key['key'] = serving_key['symbol'] + '_' + serving_key['ts'].astype(str)
            overlap_keys = set(training_key['key']).intersection(set(serving_key['key']))
            for key in list(overlap_keys)[:100]:
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
        try:
            date_filter = ""
            params: Dict[str, Any] = {}
            if start_date:
                date_filter += " AND ts >= :start_date"; params['start_date'] = start_date
            if end_date:
                date_filter += " AND ts <= :end_date"; params['end_date'] = end_date
            symbol_placeholders = ', '.join([f':sym_{i}' for i in range(len(symbols))])
            for i, symbol in enumerate(symbols):
                params[f'sym_{i}'] = symbol

            price_sql = f"""
                SELECT symbol, ts, open, close, {select_price_as('adj_close')}, volume,
                       {select_price_as('price_feat')}
                FROM daily_bars 
                WHERE symbol IN ({symbol_placeholders}) {date_filter}
                ORDER BY symbol, ts
            """
            from db import engine  # type: ignore
            prices = pd.read_sql_query(text(price_sql), engine, params=params, parse_dates=['ts'])
            if prices.empty: return pd.DataFrame()

            shares_sql = f"""
                SELECT symbol, as_of, shares
                FROM shares_outstanding 
                WHERE symbol IN ({symbol_placeholders})
                ORDER BY symbol, as_of
            """
            shares = pd.read_sql_query(text(shares_sql), engine, params=params, parse_dates=['as_of'])
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
        result_data = symbol_data[['symbol', 'ts']].copy()
        for feature_name in feature_names:
            feature_def = self.registry.get_feature(feature_name)
            if not feature_def:
                continue
            try:
                if len(symbol_data) < feature_def.lookback_days:
                    result_data[feature_name] = np.nan
                    continue
                feature_values = feature_def.computation(symbol_data)
                if isinstance(feature_values, pd.Series):
                    result_data[feature_name] = feature_values.values
                else:
                    result_data[feature_name] = feature_values
            except Exception as e:
                log.warning(f"Error computing {feature_name} for {symbol_data['symbol'].iloc[0]}: {e}")
                result_data[feature_name] = np.nan
        return result_data

    def _get_latest_date(self) -> Optional[date]:
        try:
            from db import engine  # type: ignore
            result = pd.read_sql_query(text("SELECT MAX(ts) as max_date FROM daily_bars"), engine)
            return result['max_date'].iloc[0] if not result.empty else None
        except Exception:
            return None
