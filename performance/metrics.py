import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, rf_rate: float = 0.0) -> float:
    """Compute the annualized Sharpe ratio of a return series."""
    excess = returns - rf_rate / 252.0
    # If the standard deviation is zero, return NaN to avoid divide-by-zero
    sigma = excess.std(ddof=0)
    return np.nan if sigma == 0 else np.sqrt(252) * excess.mean() / sigma


def sortino_ratio(returns: pd.Series, target: float = 0.0) -> float:
    """Compute the Sortino ratio of a return series."""
    downside = returns[returns < target]
    if downside.empty:
        return np.nan
    return (returns.mean() - target) / (downside.std(ddof=0) * np.sqrt(len(returns)))


def profit_factor(returns: pd.Series) -> float:
    """Compute the profit factor, defined as total gains divided by total losses."""
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return np.inf if losses == 0 else gains / losses


def win_rate(returns: pd.Series) -> float:
    """Compute the proportion of positive-return observations."""
    return (returns > 0).mean()


def max_drawdown(equity_curve: pd.Series) -> float:
    """Compute the maximum drawdown for an equity curve."""
    cumulative_max = equity_curve.cummax()
    drawdowns = (equity_curve - cumulative_max) / cumulative_max
    return drawdowns.min()


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Compute the one‑period Value at Risk (VaR) at a given confidence level.

    VaR at level `confidence` is defined as the percentile of the return
    distribution below which a specified proportion of observations falls.  A
    negative VaR indicates the potential loss.  This function assumes the
    returns series contains percentage returns (e.g., 0.01 for 1%).

    Parameters
    ----------
    returns : pd.Series
        A series of asset or strategy returns.
    confidence : float, optional
        The confidence level for VaR (e.g., 0.95 for 95% VaR).  Defaults to 0.95.

    Returns
    -------
    float
        The VaR estimate.  If the series is empty, returns NaN.
    """
    if returns.empty:
        return np.nan
    # compute the quantile corresponding to the (1 - confidence) tail
    return returns.quantile(1 - confidence)


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Compute the Expected Shortfall (a.k.a. Conditional VaR) at a given
    confidence level.

    Expected shortfall is the average of the returns that fall below the
    Value at Risk threshold.  Like VaR, a negative value indicates a loss.

    Parameters
    ----------
    returns : pd.Series
        A series of asset or strategy returns.
    confidence : float, optional
        The confidence level for ES (e.g., 0.95).  Defaults to 0.95.

    Returns
    -------
    float
        The Expected Shortfall estimate.  If the series is empty, returns NaN.
    """
    if returns.empty:
        return np.nan
    var_threshold = value_at_risk(returns, confidence)
    tail_losses = returns[returns <= var_threshold]
    if tail_losses.empty:
        return np.nan
    return tail_losses.mean()


def compute_all_metrics(returns: pd.Series) -> dict:
    """Compute a suite of performance metrics for a return series."""
    eq_curve = (1.0 + returns).cumprod()
    return {
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "profit_factor": profit_factor(returns),
        "win_rate": win_rate(returns),
        "max_drawdown": max_drawdown(eq_curve),
        "VaR_95": value_at_risk(returns, 0.95),
        "ES_95": expected_shortfall(returns, 0.95)
    }