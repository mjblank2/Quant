from __future__ import annotations
import numpy as np
import pandas as pd
from sqlalchemy import text
from typing import Tuple
from datetime import date, timedelta
from db import engine
from risk.sector import sector_asof
from risk.covariance import ewma_cov, robust_cov
from config import USE_FACTOR_MODEL
import logging

log = logging.getLogger(__name__)

def _load_window(start_ts: date, end_ts: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load features and prices for a time window"""
    cols = ['symbol', 'ts', 'size_ln', 'mom_21', 'turnover_21', 'beta_63']

    try:
        feats = pd.read_sql_query(
            text("SELECT {} FROM features WHERE ts>=:s AND ts<=:e".format(','.join(cols))),
            engine,
            params={'s': start_ts, 'e': end_ts},
            parse_dates=['ts']
        ).sort_values(['ts', 'symbol'])

        px = pd.read_sql_query(
            text("SELECT symbol, ts, COALESCE(adj_close, close) AS px FROM daily_bars WHERE ts>=:s AND ts<=:e"),
            engine,
            params={'s': start_ts, 'e': end_ts},
            parse_dates=['ts']
        ).sort_values(['ts', 'symbol'])

        return feats, px
    except Exception as e:
        log.error(f"Error loading data window: {e}")
        return pd.DataFrame(), pd.DataFrame()


def _build_sector_dummies(symbols: list[str], as_of: date) -> pd.DataFrame:
    """Build sector dummy variables"""
    try:
        sector_map = sector_asof(symbols, as_of)
        if not sector_map or (hasattr(sector_map, 'empty') and sector_map.empty):
            return pd.DataFrame(index=symbols)

        # Convert to DataFrame if it's a Series or dict
        if isinstance(sector_map, dict):
            sector_df = pd.Series(sector_map, name='sector')
        else:
            sector_df = sector_map

        # Create dummy variables
        sectors = sector_df.unique()
        dummies = pd.DataFrame(0, index=symbols, columns=[f'sector_{s}' for s in sectors])

        for symbol in symbols:
            if symbol in sector_df.index:
                sector = sector_df[symbol]
                dummies.loc[symbol, f'sector_{sector}'] = 1

        # Remove constant columns (sectors with only one asset)
        dummies = dummies.loc[:, dummies.std() > 1e-12]
        return dummies

    except Exception as e:
        log.warning(f"Error building sector dummies: {e}")
        return pd.DataFrame(index=symbols)


def _factor_exposure_matrix(symbols: list[str], as_of: date, feats_df: pd.DataFrame) -> pd.DataFrame:
    """Build factor exposure matrix for given symbols and date"""
    try:
        # Get features for the specific date
        df = feats_df[feats_df['ts'] == pd.to_datetime(as_of)].set_index('symbol')
        cols = [c for c in ['size_ln', 'mom_21', 'turnover_21', 'beta_63'] if c in df.columns]

        # Factor exposures
        X = df.reindex(symbols)[cols].fillna(0.0)

        # Add sector dummies
        sd = _build_sector_dummies(symbols, as_of)
        if not sd.empty:
            X = pd.concat([X, sd], axis=1)

        return X
    except Exception as e:
        log.error(f"Error building factor exposure matrix: {e}")
        return pd.DataFrame(index=symbols)


def _factor_returns(feats: pd.DataFrame, px: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Estimate factor returns using cross-sectional regression"""
    try:
        px = px.sort_values(['symbol', 'ts']).copy()

        # Calculate forward returns
        px['px_fwd'] = px.groupby('symbol')['px'].shift(-horizon)
        px['fwd_ret'] = (px['px_fwd'] / px['px']) - 1.0

        # Merge with features
        df = feats.merge(px[['symbol', 'ts', 'fwd_ret']], on=['symbol', 'ts'], how='left')
        df = df.dropna(subset=['fwd_ret'])

        if df.empty:
            return pd.DataFrame()

        cols = [c for c in ['size_ln', 'mom_21', 'turnover_21', 'beta_63'] if c in df.columns]
        out = []

        # Cross-sectional regression for each date
        for ts, g in df.groupby('ts'):
            if len(g) < 5:  # Need minimum observations
                continue

            X = g[cols].fillna(0.0).values
            y = g['fwd_ret'].values.reshape(-1, 1)

            # Add sector dummies
            sd = _build_sector_dummies(g['symbol'].tolist(), ts)
            if not sd.empty:
                X = np.concatenate([X, sd.values], axis=1)
                factor_names = cols + list(sd.columns)
            else:
                factor_names = cols

            # Add intercept
            X_ = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

            # Solve regression with regularization
            try:
                XTX = X_.T @ X_
                XTy = X_.T @ y
                beta = np.linalg.solve(XTX + 1e-6 * np.eye(XTX.shape[0]), XTy)
                fr = beta[:-1].flatten()  # Exclude intercept

                out.append(pd.Series(fr, name=ts, index=factor_names))
            except np.linalg.LinAlgError:
                continue

        if not out:
            return pd.DataFrame()

        F = pd.DataFrame(out).sort_index()
        return F

    except Exception as e:
        log.error(f"Error estimating factor returns: {e}")
        return pd.DataFrame()


def synthesize_covariance(symbols: list[str], as_of: date, lookback_days: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synthesize covariance matrix using factor model approach

    Returns:
        Tuple of (covariance_matrix, factor_exposures)
    """
    if not USE_FACTOR_MODEL or not symbols:
        # Fallback to simple covariance
        try:
            # Get historical returns
            end_date = as_of
            start_date = end_date - timedelta(days=lookback_days + 50)  # Buffer for weekends

            px_data = pd.read_sql_query(
                text("""
                SELECT symbol, ts, COALESCE(adj_close, close) AS px
                FROM daily_bars
                WHERE symbol = ANY(:syms) AND ts BETWEEN :start AND :end
                ORDER BY ts, symbol
                """),
                engine,
                params={'syms': symbols, 'start': start_date, 'end': end_date},
                parse_dates=['ts']
            )

            if px_data.empty:
                return _fallback_covariance(symbols), pd.DataFrame(index=symbols)

            # Calculate returns
            returns = px_data.pivot(index='ts', columns='symbol', values='px').pct_change().dropna()
            returns = returns.reindex(columns=symbols).tail(lookback_days)

            cov_matrix = robust_cov(returns, method='ewma')
            return cov_matrix, pd.DataFrame(index=symbols)

        except Exception as e:
            log.error(f"Error in fallback covariance: {e}")
            return _fallback_covariance(symbols), pd.DataFrame(index=symbols)

    try:
        # Load data window
        end_date = as_of
        start_date = end_date - timedelta(days=lookback_days + 50)
        feats, px = _load_window(start_date, end_date)

        if feats.empty or px.empty:
            return _fallback_covariance(symbols), pd.DataFrame(index=symbols)

        # Build factor exposure matrix for current date
        B = _factor_exposure_matrix(symbols, as_of, feats)

        if B.empty:
            return _fallback_covariance(symbols), B

        # Estimate factor returns time series
        F = _factor_returns(feats, px)

        if F.empty or len(F) < 20:
            # Not enough data for factor model
            return _fallback_covariance(symbols), B

        # Factor covariance matrix
        F_cov = robust_cov(F, method='ewma')

        # Specific risk (residual risk)
        # For simplicity, assume diagonal specific risk
        px_pivot = px.pivot(index='ts', columns='symbol', values='px')
        returns = px_pivot.pct_change().dropna()
        returns = returns.reindex(columns=symbols).tail(lookback_days)

        if returns.empty:
            specific_var = pd.Series(1e-4, index=symbols)
        else:
            # Estimate specific variance as residual variance after factor model
            specific_var = returns.var() * 0.5  # Rough estimate: half of total variance is specific
            specific_var = specific_var.fillna(1e-4).clip(lower=1e-6)

        # Synthesize covariance: Cov = B * F_cov * B' + Specific_Risk
        B_aligned = B.reindex(index=symbols, columns=F_cov.index).fillna(0.0)
        factor_cov = B_aligned @ F_cov @ B_aligned.T

        # Add specific risk on diagonal
        specific_risk = pd.DataFrame(np.diag(specific_var),
                                   index=symbols, columns=symbols)

        total_cov = factor_cov + specific_risk

        # Ensure positive definiteness
        eigenvals, eigenvecs = np.linalg.eigh(total_cov.values)
        eigenvals = np.maximum(eigenvals, 1e-8)
        cov_clean = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        total_cov = pd.DataFrame(cov_clean, index=symbols, columns=symbols)

        return total_cov, B

    except Exception as e:
        log.error(f"Error synthesizing covariance: {e}")
        return _fallback_covariance(symbols), pd.DataFrame(index=symbols)


def _fallback_covariance(symbols: list[str]) -> pd.DataFrame:
    """Fallback diagonal covariance matrix"""
    n = len(symbols)
    if n == 0:
        return pd.DataFrame()

    # Simple diagonal matrix with reasonable variance
    cov_matrix = pd.DataFrame(np.eye(n) * 1e-4, index=symbols, columns=symbols)
    return cov_matrix
