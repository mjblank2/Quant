"""
Performance metrics for evaluating trading strategies.

This module provides functions for computing standard and adjusted Sharpe
ratios as well as probabilistic and deflated Sharpe ratios (PSR and DSR).
These metrics help assess the statistical significance of a strategy's
performance while accounting for skewness, kurtosis, sample size, and
multiple testing. The formulas are adapted from M. López de Prado's
*Advances in Financial Machine Learning* and related academic research.

Functions
---------
sharpe_ratio(returns) -> float
    Compute the mean divided by the standard deviation of a return series.

annualized_sharpe(returns, periods_per_year) -> float
    Scale the Sharpe ratio to an annualized figure assuming a given
    number of return observations per year.

psr(returns, sr_benchmark) -> float
    Compute the probabilistic Sharpe ratio, i.e. the probability that
    the true Sharpe ratio exceeds a benchmark.

dsr(returns, candidate_srs) -> float
    Compute the deflated Sharpe ratio, adjusting for multiple testing
    by estimating an expected maximum Sharpe from a set of candidates.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

def sharpe_ratio(returns: pd.Series) -> float:
    """Compute the (periodic) Sharpe ratio of a returns series.

    Parameters
    ----------
    returns : pd.Series
        Series of returns (periodic, not annualized) with frequency
        consistent across the sample.

    Returns
    -------
    float
        The Sharpe ratio (mean divided by std). If the standard
        deviation is zero or the series is empty, returns zero.
    """
    if returns is None or returns.empty:
        return 0.0
    clean = returns.dropna().astype(float)
    if clean.empty:
        return 0.0
    mu = clean.mean()
    sigma = clean.std(ddof=0)
    return 0.0 if sigma == 0 else mu / sigma


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualize the Sharpe ratio given a sampling frequency.

    Parameters
    ----------
    returns : pd.Series
        Series of periodic returns.
    periods_per_year : int, default 252
        Number of return observations per year (e.g., 252 for daily).

    Returns
    -------
    float
        The annualized Sharpe ratio.
    """
    sr = sharpe_ratio(returns)
    return sr * np.sqrt(periods_per_year)


def _variance_of_sharpe(sr: float, n: int) -> float:
    """Estimate the variance of the Sharpe ratio under the null hypothesis.
    Approximate formula: Var(SR) ≈ (1 + SR^2 / 2) / (n - 1).
    """
    return (1.0 + (sr**2) / 2.0) / max(1, n - 1)


def psr(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """Compute the Probabilistic Sharpe Ratio (PSR).

    The PSR is the probability that the true Sharpe ratio exceeds
    a benchmark Sharpe ratio given the observed sample Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Series of returns (periodic).
    sr_benchmark : float, default 0.0
        Benchmark Sharpe ratio to test against.

    Returns
    -------
    float
        The probabilistic Sharpe ratio (0–1). If the variance of the
        Sharpe estimate is undefined due to insufficient data, returns 0.5.
    """
    clean = returns.dropna().astype(float)
    n = len(clean)
    if n < 2:
        return 0.5  # Indeterminate
    sr = sharpe_ratio(clean)
    var_hat = _variance_of_sharpe(sr, n)
    if var_hat <= 0:
        return 0.5
    from math import erf, sqrt
    # z-score for SR difference
    z = (sr - sr_benchmark) / np.sqrt(var_hat)
    # PSR = CDF of z
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def dsr(returns: pd.Series, candidate_srs: Iterable[float]) -> float:
    """Compute the Deflated Sharpe Ratio (DSR) accounting for multiple testing.

    Parameters
    ----------
    returns : pd.Series
        Series of returns (periodic).
    candidate_srs : Iterable[float]
        Sharpe ratios of all candidate strategies tested. The DSR adjusts
        for the fact that selecting the best among many can inflate the
        observed Sharpe ratio.

    Returns
    -------
    float
        The deflated Sharpe ratio (0–1). Returns 0.5 if insufficient data.
    """
    clean = returns.dropna().astype(float)
    n = len(clean)
    if n < 2:
        return 0.5
    sr_obs = sharpe_ratio(clean)
    # Estimate variance of Sharpe under null
    var_hat = _variance_of_sharpe(sr_obs, n)
    if var_hat <= 0:
        return 0.5
    candidate_srs = list(candidate_srs)
    if not candidate_srs:
        return psr(clean, sr_benchmark=0.0)
    m = len(candidate_srs)
    from math import sqrt, log, erf
    # Approximate expected maximum SR for m normal variables
    gamma = 0.57721566490153286060  # Euler-Mascheroni constant
    expected_max = np.sqrt(var_hat) * (
        (1.0 - gamma) * np.sqrt(2.0 * np.log(m)) + gamma * np.sqrt(2.0 * np.log(m / np.e))
    )
    z = (sr_obs - expected_max) / np.sqrt(var_hat)
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))