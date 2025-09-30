"""
Performance and risk metrics for trading strategies.

This module extends the original metrics implementation with additional
risk-adjusted measures that are commonly used by professional traders and
quantitative funds.  In addition to Sharpe, Sortino, profit factor, win rate,
maximum drawdown and tail-risk metrics (VaR and expected shortfall), the
functions below implement the Calmar ratio, Omega ratio, skewness and
kurtosis.  These metrics provide deeper insights into the return
distribution, capturing both the magnitude and the shape of gains and
losses.

The Calmar ratio compares the annualized rate of return to the maximum
drawdown, emphasising downside risk.  A higher Calmar indicates that the
strategy delivers strong returns relative to its worst drawdown.  The Omega
ratio measures how much more the strategy profits above a chosen threshold
than it loses below that threshold.  A value above one implies the
strategy’s upside outweighs its downside.  Skewness and kurtosis describe
the asymmetry and tail heaviness of returns, respectively; they can be
useful for understanding whether the strategy is prone to extreme events.
"""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, rf_rate: float = 0.0) -> float:
    """Compute the annualized Sharpe ratio of a return series."""
    excess = returns - rf_rate / 252.0
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
    distribution below which a specified proportion of observations falls.
    A negative VaR indicates the potential loss.  This function assumes the
    returns series contains percentage returns (e.g., 0.01 for 1%).
    """
    if returns.empty:
        return np.nan
    return returns.quantile(1 - confidence)


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Compute the Expected Shortfall (a.k.a. Conditional VaR) at a given
    confidence level.  Expected shortfall is the average of the returns that
    fall below the VaR threshold.  Like VaR, a negative value indicates a
    loss.
    """
    if returns.empty:
        return np.nan
    var_threshold = value_at_risk(returns, confidence)
    tail_losses = returns[returns <= var_threshold]
    if tail_losses.empty:
        return np.nan
    return tail_losses.mean()


def calmar_ratio(returns: pd.Series) -> float:
    """
    Compute the Calmar ratio of a return series.

    The Calmar ratio is the annualized return divided by the absolute value
    of the maximum drawdown.  It penalizes strategies that experience large
    drawdowns relative to their gains.  Returns are assumed to be daily.
    """
    if returns.empty:
        return np.nan
    # Cumulative equity curve
    eq = (1.0 + returns).cumprod()
    # Total return and CAGR
    total_return = eq.iloc[-1] - 1.0
    years = len(returns) / 252.0
    if years <= 0:
        return np.nan
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    mdd = max_drawdown(eq)
    # Avoid division by zero; note mdd is negative or zero
    if mdd >= 0 or np.isclose(mdd, 0.0):
        return np.nan
    return cagr / abs(mdd)


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Compute the Omega ratio of a return series.

    The Omega ratio is the ratio of the expected gains above a threshold to the
    expected losses below that threshold.  A value greater than 1 implies
    favourable upside relative to downside.  If there are no losses, the
    function returns infinity.
    """
    if returns.empty:
        return np.nan
    # Excess returns relative to the threshold
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    sum_gains = gains.sum()
    sum_losses = losses.sum()
    # If no downside, return infinity; if no upside, return zero
    if sum_losses == 0:
        return np.inf if sum_gains > 0 else 0.0
    return sum_gains / sum_losses


def skewness(returns: pd.Series) -> float:
    """Return the sample skewness of the return series."""
    return returns.skew() if not returns.empty else np.nan


def kurtosis(returns: pd.Series) -> float:
    """Return the sample kurtosis of the return series."""
    return returns.kurtosis() if not returns.empty else np.nan


def roi(returns: pd.Series) -> float:
    """Return on investment for the return series.

    This metric computes the total return over the period.  For a daily
    return series, ROI is simply the cumulative product minus one.  A
    positive value indicates profit; negative indicates loss.
    """
    if returns.empty:
        return np.nan
    cumulative = (1.0 + returns).cumprod()
    return cumulative.iloc[-1] - 1.0


def cagr(returns: pd.Series) -> float:
    """Compute the Compound Annual Growth Rate (CAGR).

    CAGR measures the mean annual growth rate of an investment over a period
    of time, assuming profits are reinvested.  For a daily return series,
    the formula is:

    .. code-block:: python

        eq_end = (1 + returns).prod()
        years = len(returns) / 252.0
        cagr = eq_end ** (1 / years) - 1

    If the series is empty or the horizon is zero, returns NaN.
    """
    if returns.empty:
        return np.nan
    eq_end = (1.0 + returns).prod()
    years = len(returns) / 252.0
    if years <= 0:
        return np.nan
    return eq_end ** (1.0 / years) - 1.0


def treynor_ratio(returns: pd.Series, market_returns: pd.Series, rf_rate: float = 0.0) -> float:
    """Compute the Treynor ratio for a strategy.

    The Treynor ratio evaluates performance relative to systematic risk
    (beta).  It is defined as::

        (mean(returns) - rf_rate / 252) / beta

    where beta is the slope of the linear regression of the strategy
    returns against market_returns.  This function assumes both series are
    aligned and of equal length.  If beta cannot be computed (e.g., zero
    variance in the market), it returns NaN.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy.
    market_returns : pd.Series
        Daily returns of the benchmark or market index.
    rf_rate : float, default 0.0
        Annualized risk‑free rate.  The daily rate is rf_rate/252.

    Returns
    -------
    float
        The Treynor ratio.
    """
    if returns.empty or market_returns.empty:
        return np.nan
    # Align on index intersection
    aligned = pd.concat([returns, market_returns], axis=1, join='inner').dropna()
    if aligned.empty:
        return np.nan
    r = aligned.iloc[:, 0] - rf_rate / 252.0
    m = aligned.iloc[:, 1]
    # Compute beta: covariance(m, r) / var(m)
    var_m = m.var(ddof=0)
    if var_m == 0 or np.isnan(var_m):
        return np.nan
    beta = m.cov(r) / var_m
    if beta == 0 or np.isnan(beta):
        return np.nan
    return r.mean() / beta


# -----------------------------------------------------------------------------
# Risk‑adjusted performance metrics requiring benchmark returns
# -----------------------------------------------------------------------------

def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Compute the Information Ratio (IR) of a strategy.

    The information ratio compares the average excess return of the portfolio to
    the volatility of those excess returns (tracking error) against a
    benchmark.  It is defined as:

    .. math::

       \text{IR} = \frac{\mu_{\text{active}}}{\sigma_{\text{active}}},

    where ``mu_active`` is the mean of ``returns - benchmark_returns`` and
    ``sigma_active`` is the standard deviation of ``returns - benchmark_returns``.
    A higher IR indicates that the strategy is generating excess returns
    consistently relative to the benchmark【901753389139961†L259-L297】.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy.
    benchmark_returns : pd.Series
        Daily returns of the benchmark index.

    Returns
    -------
    float
        The information ratio, or NaN if the inputs are insufficient.
    """
    if returns.empty or benchmark_returns.empty:
        return np.nan
    aligned = pd.concat([returns, benchmark_returns], axis=1, join='inner').dropna()
    if aligned.empty:
        return np.nan
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    tracking_error = active.std(ddof=0)
    if tracking_error == 0 or np.isnan(tracking_error):
        return np.nan
    return active.mean() / tracking_error


def tail_ratio(returns: pd.Series, high: float = 0.95, low: float = 0.05) -> float:
    """Compute the tail ratio of a return series.

    The tail ratio is the ratio of the magnitude of extreme gains to extreme
    losses.  It is defined as::

        tail_ratio = percentile(returns, high) / abs(percentile(returns, low))

    where ``high`` and ``low`` denote upper and lower quantile levels (defaults
    to 95th and 5th percentiles).  A tail ratio greater than 1 indicates that
    the largest gains outweigh the largest losses, while a ratio below 1
    indicates that losses dominate.

    Parameters
    ----------
    returns : pd.Series
        Return series.
    high : float, default 0.95
        Upper quantile for extreme gains.
    low : float, default 0.05
        Lower quantile for extreme losses.

    Returns
    -------
    float
        The tail ratio, or NaN if the series is empty or the lower percentile
        is zero.
    """
    if returns.empty:
        return np.nan
    upper = returns.quantile(high)
    lower = returns.quantile(low)
    if lower >= 0 or np.isclose(lower, 0.0):
        return np.nan
    return upper / abs(lower)


def ulcer_index(equity_curve: pd.Series) -> float:
    """Compute the Ulcer Index of an equity curve.

    The Ulcer Index measures the depth and duration of drawdowns by taking the
    square root of the mean squared percentage drawdowns.  It is defined as::

        UI = sqrt( mean( drawdown^2 ) )

    where drawdown is computed relative to the running maximum of the equity
    curve.  Lower values indicate smoother equity growth, while higher values
    indicate prolonged or severe drawdowns.

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity series (e.g., cumulative product of returns).

    Returns
    -------
    float
        The Ulcer Index, or NaN if the series is empty.
    """
    if equity_curve.empty:
        return np.nan
    running_max = equity_curve.cummax()
    dd = (equity_curve - running_max) / running_max
    return np.sqrt((dd ** 2).mean())


def gain_to_pain_ratio(returns: pd.Series) -> float:
    """Compute the gain-to-pain ratio of a return series.

    The gain-to-pain ratio (GPR) divides the cumulative return by the sum
    of absolute losses.  Unlike the profit factor, which compares total
    gains to total losses, GPR compares total net profit to the total pain
    endured (absolute losses).  It is defined as::

        GPR = (sum(returns)) / sum(|returns| where returns < 0)

    If there are no losing periods, the function returns infinity.

    Parameters
    ----------
    returns : pd.Series
        Return series.

    Returns
    -------
    float
        Gain-to-pain ratio, or NaN if the series is empty.
    """
    if returns.empty:
        return np.nan
    losses = returns[returns < 0].abs().sum()
    if losses == 0:
        return np.inf
    return returns.sum() / losses


def compute_all_metrics(returns: pd.Series, market_returns: pd.Series | None = None) -> dict:
    """
    Compute a suite of performance metrics for a return series.

    This function aggregates traditional risk‑adjusted performance metrics
    alongside additional measures of tail risk, distribution shape and
    investment growth.  It assumes ``returns`` is a daily return series and
    will annualize where appropriate.  When ``market_returns`` is supplied,
    the Treynor ratio and information ratio will be computed using the
    strategy's returns relative to the benchmark; otherwise they are set to
    ``NaN``.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy.
    market_returns : pd.Series or None
        Daily returns of the benchmark index (optional).  When provided,
        additional metrics (Treynor and information ratio) are computed.

    Returns
    -------
    dict
        Dictionary mapping metric names to values.
    """
    eq_curve = (1.0 + returns).cumprod()
    metrics = {
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "profit_factor": profit_factor(returns),
        "win_rate": win_rate(returns),
        "max_drawdown": max_drawdown(eq_curve),
        "VaR_95": value_at_risk(returns, 0.95),
        "ES_95": expected_shortfall(returns, 0.95),
        "ES_99": expected_shortfall(returns, 0.99),
        "calmar": calmar_ratio(returns),
        "omega": omega_ratio(returns),
        "skew": skewness(returns),
        "kurtosis": kurtosis(returns),
        "roi": roi(returns),
        "cagr": cagr(returns),
        "tail_ratio": tail_ratio(returns),
        "ulcer_index": ulcer_index(eq_curve),
        "gain_to_pain": gain_to_pain_ratio(returns),
    }
    # Additional ratios requiring benchmark returns
    if market_returns is not None:
        metrics["treynor"] = treynor_ratio(returns, market_returns)
        metrics["information_ratio"] = information_ratio(returns, market_returns)
    else:
        metrics["treynor"] = np.nan
        metrics["information_ratio"] = np.nan
    return metrics