import numpy as np

def sharpe_ratio(returns, rf_rate=0.0):
    excess = returns - rf_rate / 252.0
    return np.sqrt(252) * excess.mean() / excess.std(ddof=0)

def sortino_ratio(returns, target=0.0):
    downside = returns[returns < target]
    if downside.empty:
        return np.nan
    return (returns.mean() - target) / (downside.std(ddof=0) * np.sqrt(len(returns)))

def profit_factor(returns):
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return np.inf if losses == 0 else gains / losses

def win_rate(returns):
    return (returns > 0).mean()

def max_drawdown(equity_curve):
    cumulative_max = equity_curve.cummax()
    drawdowns = (equity_curve - cumulative_max) / cumulative_max
    return drawdowns.min()

def compute_all_metrics(returns):
    eq_curve = (1.0 + returns).cumprod()
    return {
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "profit_factor": profit_factor(returns),
        "win_rate": win_rate(returns),
        "max_drawdown": max_drawdown(eq_curve)
    }
