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


def compute_all_metrics(returns: pd.Series) -> dict:
    """Compute a suite of performance metrics for a return series."""
    eq_curve = (1.0 + returns).cumprod()
    return {
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "profit_factor": profit_factor(returns),
        "win_rate": win_rate(returns),
        "max_drawdown": max_drawdown(eq_curve)
    }