from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from config import EWMA_LAMBDA, USE_LEDOIT_WOLF

def ewma_cov(returns: pd.DataFrame, lam: float | None = None) -> pd.DataFrame:
    """
    Exponentially Weighted Moving Average covariance estimator

    Args:
        returns: DataFrame of asset returns (dates x assets)
        lam: Decay factor (default from config)

    Returns:
        Covariance matrix as DataFrame
    """
    lam = lam if lam is not None else EWMA_LAMBDA
    R = returns.dropna(how='all').fillna(0.0).values

    if R.shape[0] < 2:
        return pd.DataFrame(np.eye(R.shape[1]), index=returns.columns, columns=returns.columns)

    # EWMA calculation - more recent observations have higher weight
    S = np.zeros((R.shape[1], R.shape[1]))
    total = 0.0

    # Iterate backwards through time (most recent first)
    for t in range(R.shape[0]-1, -1, -1):
        x = R[t:t+1].T  # Column vector
        S = lam * S + (1 - lam) * (x @ x.T)
        total = lam * total + (1 - lam)

    S = S / max(total, 1e-8)
    return pd.DataFrame(S, index=returns.columns, columns=returns.columns)


def shrink_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Shrinkage covariance estimator (Ledoit-Wolf or sample covariance)

    Args:
        returns: DataFrame of asset returns

    Returns:
        Shrunk covariance matrix
    """
    R = returns.dropna(how='all').fillna(0.0).values

    if R.shape[0] < 2:
        return pd.DataFrame(np.eye(R.shape[1]), index=returns.columns, columns=returns.columns)

    if USE_LEDOIT_WOLF:
        try:
            lw = LedoitWolf().fit(R)
            S = lw.covariance_
        except Exception:
            # Fallback to sample covariance if Ledoit-Wolf fails
            S = np.cov(R, rowvar=False)
    else:
        S = np.cov(R, rowvar=False)

    return pd.DataFrame(S, index=returns.columns, columns=returns.columns)


def robust_cov(returns: pd.DataFrame, method: str = 'ewma') -> pd.DataFrame:
    """
    Robust covariance estimation with fallback options

    Args:
        returns: DataFrame of asset returns
        method: 'ewma', 'shrink', or 'sample'

    Returns:
        Covariance matrix
    """
    if returns.empty or returns.shape[0] < 2:
        n_assets = len(returns.columns) if not returns.empty else 1
        return pd.DataFrame(np.eye(n_assets) * 1e-4,
                          index=returns.columns if not returns.empty else ['DUMMY'],
                          columns=returns.columns if not returns.empty else ['DUMMY'])

    try:
        if method == 'ewma':
            return ewma_cov(returns)
        elif method == 'shrink':
            return shrink_cov(returns)
        else:  # sample
            R = returns.dropna(how='all').fillna(0.0)
            cov_matrix = R.cov()
            # Ensure positive definiteness
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix.values)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Floor eigenvalues
            cov_clean = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            return pd.DataFrame(cov_clean, index=cov_matrix.index, columns=cov_matrix.columns)

    except Exception:
        # Ultimate fallback: diagonal covariance
        return pd.DataFrame(np.eye(len(returns.columns)) * returns.var().mean(),
                            index=returns.columns, columns=returns.columns)
