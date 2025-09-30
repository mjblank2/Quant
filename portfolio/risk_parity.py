"""
Portfolio construction tools for risk parity and target volatility scaling.

This module provides helper functions to compute risk parity weights and to
rescale a portfolio to a desired annualised volatility.  Risk parity aims to
allocate capital such that each position contributes an equal share of the
portfolio's total risk.  The algorithms here are deliberately simple and
lightweight so they can be run inside the existing infrastructure without
additional dependencies.

Example usage:

```
import pandas as pd
from portfolio.risk_parity import risk_parity_weights, target_volatility_scaling

# Suppose we have a covariance matrix for 3 assets
cov = pd.DataFrame(
    [[0.04, 0.02, 0.01],
     [0.02, 0.09, 0.03],
     [0.01, 0.03, 0.16]],
    index=['A','B','C'], columns=['A','B','C']
)

weights = risk_parity_weights(cov)
# Rescale to target annual volatility of 15%
scaled = target_volatility_scaling(weights, cov, target_vol=0.15)
```

The functions return pandas Series indexed by asset name for easy alignment
with other data structures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

def risk_parity_weights(cov: pd.DataFrame, max_iter: int = 1000, tol: float = 1e-6) -> pd.Series:
    """Compute risk parity weights for a positive semi-definite covariance matrix.

    Risk parity seeks weights such that each asset contributes an equal share to
    total portfolio variance.  This implementation uses an iterative procedure
    similar to the one described by Maillard, Roncalli and Teiletche (2010).

    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix of asset returns.  The index and columns should be
        identical and represent the asset identifiers.  The matrix must be
        positive semi-definite; however, small numerical noise will be handled
        gracefully.
    max_iter : int, default 1000
        Maximum number of iterations to perform.
    tol : float, default 1e-6
        Convergence tolerance on the L1 norm of risk contributions.  The
        algorithm stops when the sum of absolute differences between each
        asset's risk contribution and the average risk contribution falls
        below this threshold.

    Returns
    -------
    pd.Series
        Risk parity weights indexed by the asset identifiers.  The weights sum
        to one and are non-negative.
    """
    # Ensure input is a DataFrame
    if not isinstance(cov, pd.DataFrame):
        cov = pd.DataFrame(cov)
    assets = cov.index
    n = len(assets)
    if n == 0:
        return pd.Series(dtype=float)
    # Start from equal weights
    w = np.full(n, 1.0 / n)
    cov_matrix = cov.to_numpy()
    for _ in range(max_iter):
        # Compute portfolio variance and marginal contributions
        port_var = float(w @ cov_matrix @ w)
        if port_var <= 0:
            break
        # Marginal contribution of each asset to portfolio variance
        mrc = cov_matrix @ w
        # Risk contributions (percentage of total variance)
        rc = w * mrc
        # Desired risk contributions: equal for all assets
        target_rc = port_var / n
        # Error on contributions
        diff = rc - target_rc
        if np.abs(diff).sum() < tol:
            break
        # Update weights: scale by ratio of target to actual contribution
        # Add a small constant to the denominator to avoid division by zero
        adjustment = target_rc / (rc + 1e-12)
        w *= adjustment
        # Re-normalise weights to sum to one and enforce positivity
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s > 0:
            w /= s
        else:
            w = np.full(n, 1.0 / n)
    return pd.Series(w, index=assets)


def target_volatility_scaling(weights: pd.Series, cov: pd.DataFrame, target_vol: float = 0.1) -> pd.Series:
    """Rescale a portfolio to hit a desired annualised volatility target.

    The function computes the annualised volatility of the given portfolio
    (assuming daily returns and 252 trading days per year) and scales the
    weights so that the resulting portfolio volatility matches the target.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights indexed by asset identifier.  These may sum to any
        value; only their relative magnitudes are used for scaling.
    cov : pd.DataFrame
        Covariance matrix of asset returns.  Index and columns should match
        the weights' index.
    target_vol : float, default 0.1
        Desired annualised volatility.  For example, 0.1 corresponds to 10%
        annualised volatility.

    Returns
    -------
    pd.Series
        Rescaled weights that aim to achieve the target annualised volatility.
    """
    # Align covariance matrix with weights
    cov = cov.loc[weights.index, weights.index]
    # Compute current annualised volatility
    port_var = float(weights.to_numpy() @ cov.to_numpy() @ weights.to_numpy())
    port_vol = np.sqrt(port_var) * np.sqrt(252.0)  # annualise
    if port_vol <= 0 or np.isnan(port_vol):
        return weights
    # Scale factor to reach target volatility
    scale = target_vol / port_vol
    return weights * scale


def build_portfolio_risk_parity(alpha: pd.Series, features: pd.DataFrame, target_vol: float = 0.1) -> pd.Series:
    """Construct a long-only portfolio using risk parity weights.

    This function takes alpha scores and corresponding features (including
    volatility estimates) to compute a covariance matrix.  It then derives
    risk parity weights, rescales them to match the specified volatility
    target, and returns the resulting weight vector.

    Parameters
    ----------
    alpha : pd.Series
        Series of expected returns or alpha scores indexed by symbol.
    features : pd.DataFrame
        Feature DataFrame indexed by symbol.  Must include a 'vol_63' column
        representing 63-day realised volatility.  The covariance matrix is
        approximated as diag(vol_63**2) since cross-asset covariances are
        unknown at trade time.  If a more sophisticated estimator is desired,
        users can replace this logic with their own covariance calculation.
    target_vol : float, default 0.1
        Desired annualised volatility for the resulting portfolio.

    Returns
    -------
    pd.Series
        Long-only weights indexed by symbol.  Weights are scaled to meet the
        target volatility and sum to the gross leverage (assumed to be 1
        before scaling).  Negative weights are clipped to zero.
    """
    if alpha.empty:
        return pd.Series(dtype=float)
    syms = alpha.index.tolist()
    vols = features.reindex(syms)['vol_63'].abs().replace(0, np.nan)
    # Build a diagonal covariance matrix using vol_63 (daily) squared
    # Convert to daily variance by dividing annualised vol by sqrt(252)
    daily_vols = vols / np.sqrt(252.0)
    cov = pd.DataFrame(np.diag(daily_vols ** 2), index=syms, columns=syms)
    # Risk parity weights from covariance
    w = risk_parity_weights(cov)
    # Scale by alpha ranking: multiply by normalized alpha to tilt weights
    # We normalise alpha to have unit sum of absolute values
    a = alpha.abs()
    a_sum = a.sum()
    if a_sum > 0:
        tilt = a / a_sum
        # Combine risk parity baseline with alpha tilt
        w = w * tilt
        s = w.sum()
        if s > 0:
            w = w / s
    # Ensure no negative weights
    w = w.clip(lower=0.0)
    # Rescale to target volatility
    w = target_volatility_scaling(w, cov, target_vol)
    # Normalise to sum to target gross leverage (1.0) before returning
    total = w.sum()
    if total > 0:
        w = w / total
    return w