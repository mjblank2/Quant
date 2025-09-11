
from __future__ import annotations

import logging
from datetime import date
from typing import Dict

import numpy as np
import pandas as pd

try:
    import cvxpy as cp  # type: ignore
except Exception:
    cp = None

from sqlalchemy import text, bindparam
from db import engine  # type: ignore
from utils.price_utils import select_price_as
try:
    from config import (
        GROSS_LEVERAGE, NET_EXPOSURE, MAX_POSITION_WEIGHT, MIN_PRICE, MIN_ADV_USD,
        MVO_RISK_LAMBDA, MVO_COST_LAMBDA, BETA_MIN, BETA_MAX, TURNOVER_LIMIT_ANNUAL,
        LIQUIDITY_MAX_PCT_ADV, STARTING_CAPITAL
    )
except Exception:
    GROSS_LEVERAGE = 1.0; NET_EXPOSURE = 1.0; MAX_POSITION_WEIGHT = 0.10; MIN_PRICE = 1.0; MIN_ADV_USD = 100000.0
    MVO_RISK_LAMBDA = 10.0; MVO_COST_LAMBDA = 0.0; BETA_MIN = 0.0; BETA_MAX = 2.0; TURNOVER_LIMIT_ANNUAL = 5.0
    LIQUIDITY_MAX_PCT_ADV = 0.05; STARTING_CAPITAL = 1_000_000.0

try:
    from risk.risk_model import est_beta_asof, neutralize_with_sectors  # type: ignore
except Exception:
    def est_beta_asof(syms, as_of, market_symbol="IWM", lookback=63):
        return pd.Series(1.0, index=syms)
try:
    from risk.factor_model import synthesize_covariance  # type: ignore
except Exception:
    def synthesize_covariance(syms, as_of):
        n = len(syms)
        Sigma = pd.DataFrame(np.eye(n)*1e-4, index=syms, columns=syms)
        return Sigma, pd.DataFrame(index=syms)

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dynamic risk lambda helper
# -----------------------------------------------------------------------------
def _calculate_dynamic_risk_lambda(as_of: date, base_lambda: float, market_symbol: str = "IWM", target_vol: float = 0.15) -> float:
    """
    Adjust the risk aversion parameter based on recent market volatility.

    Parameters
    ----------
    as_of : date
        Date for which to calculate the risk aversion.  Typically the portfolio
        date.  The lookback period is anchored off this date.
    base_lambda : float
        Baseline risk aversion parameter from configuration.
    market_symbol : str, default "IWM"
        The symbol used to estimate market volatility.
    target_vol : float, default 0.15
        Desired target volatility (annualized).  If realized volatility exceeds
        this, risk aversion increases; if lower, risk aversion decreases.

    Returns
    -------
    float
        Adjusted risk aversion parameter.  The adjustment factor is constrained
        between 0.5x and 3x of the base lambda.
    """
    lookback = 63  # use roughly 3 months of data
    start_date = as_of - pd.Timedelta(days=int(lookback * 1.5))
    # Query market prices
    sql = text(f"SELECT ts, {select_price_as('px')} FROM daily_bars WHERE symbol=:s AND ts>=:start AND ts<=:end ORDER BY ts")
    try:
        df = pd.read_sql_query(sql, engine, params={'s': market_symbol, 'start': start_date, 'end': as_of}, parse_dates=['ts'])
    except Exception:
        return base_lambda
    if df.empty or len(df) < lookback:
        return base_lambda
    returns = df['px'].pct_change()
    realized_vol = returns.rolling(window=lookback).std().iloc[-1] * np.sqrt(252)
    if pd.isna(realized_vol) or realized_vol == 0:
        return base_lambda
    adj_factor = realized_vol / target_vol
    dynamic_lambda = base_lambda * adj_factor
    dynamic_lambda = max(base_lambda * 0.5, min(base_lambda * 3.0, dynamic_lambda))
    log.info(f"Market Vol: {realized_vol:.2f}. Base Lambda: {base_lambda:.2f}. Dynamic Lambda: {dynamic_lambda:.2f}")
    return dynamic_lambda

def _latest_prices(symbols: list[str]) -> pd.Series:
    """
    Fetch the latest price for each symbol in the list.

    The query uses a subquery to select the maximum timestamp per symbol and then
    joins back to daily_bars.  Symbols are passed via an expanded bindparam to
    support lists of arbitrary length without using ANY(), which may not be
    portable across all SQL dialects.
    """
    if not symbols:
        return pd.Series(dtype=float)
    stmt = text(f"""
        WITH latest AS (
            SELECT symbol, MAX(ts) AS ts
            FROM daily_bars
            WHERE symbol IN :syms
            GROUP BY symbol
        )
        SELECT b.symbol, {select_price_as('px')}
        FROM daily_bars b
        JOIN latest l ON b.symbol = l.symbol AND b.ts = l.ts
    """).bindparams(bindparam("syms", expanding=True))
    try:
        df = pd.read_sql_query(stmt, engine, params={'syms': tuple(symbols)})
        return df.set_index("symbol")["px"] if not df.empty else pd.Series(dtype=float)
    except Exception as e:
        log.error(f"Error fetching prices: {e}")
        return pd.Series(dtype=float)

def _adv20(symbols: list[str]) -> pd.Series:
    if not symbols:
        return pd.Series(dtype=float)
    stmt = text("SELECT symbol, adv_usd_20 FROM universe WHERE symbol IN :syms")
    stmt = stmt.bindparams(bindparam("syms", expanding=True))
    try:
        df = pd.read_sql_query(stmt, engine, params={'syms': tuple(symbols)})
        return df.set_index("symbol")["adv_usd_20"] if not df.empty else pd.Series(dtype=float)
    except Exception as e:
        log.error(f"Error fetching ADV: {e}")
        return pd.Series(dtype=float)

def _load_previous_weights(symbols: list[str]) -> pd.Series:
    """
    Load previous target weights for given symbols from the most recent timestamp.
    Uses an IN-clause with expanded parameters to ensure portability.
    """
    if not symbols:
        return pd.Series(dtype=float)
    stmt = text("""
        SELECT symbol, weight
        FROM target_positions
        WHERE ts = (SELECT MAX(ts) FROM target_positions)
        AND symbol IN :syms
    """).bindparams(bindparam("syms", expanding=True))
    try:
        df = pd.read_sql_query(stmt, engine, params={'syms': tuple(symbols)})
        return df.set_index('symbol')['weight'] if not df.empty else pd.Series(dtype=float)
    except Exception as e:
        log.warning(f"Error loading previous weights: {e}")
        return pd.Series(dtype=float)

def build_portfolio_mvo(alpha: pd.Series, as_of: date, risk_lambda: float | None = None) -> pd.Series:
    # Fall back to heuristic optimizer if CVXPY is not available
    if cp is None:
        log.warning("CVXPY not available, falling back to simple optimization")
        return _fallback_optimizer(alpha)

    # Determine effective risk aversion (dynamic based on market volatility)
    base_lambda = risk_lambda if risk_lambda is not None else MVO_RISK_LAMBDA
    effective_lambda = _calculate_dynamic_risk_lambda(as_of, base_lambda)

    syms = alpha.index.tolist()
    px = _latest_prices(syms)
    adv = _adv20(syms)

    # Filter out symbols that fail price and liquidity screens
    ok_mask = (px >= MIN_PRICE) & (adv >= MIN_ADV_USD)
    a = alpha[ok_mask].fillna(0.0)
    if a.empty:
        log.warning("No symbols passed filters")
        return pd.Series(dtype=float)
    syms = a.index.tolist()
    px = px.reindex(syms)
    adv = adv.reindex(syms)

    # Build risk model
    log.info(f"Building risk model for {len(syms)} symbols")
    Sigma, factor_exposures = synthesize_covariance(syms, as_of)
    if Sigma.empty:
        log.warning("Failed to build risk model, using diagonal covariance")
        Sigma = pd.DataFrame(np.eye(len(syms)) * 1e-4, index=syms, columns=syms)

    # Ensure covariance matrix is positive definite; regularize if necessary
    try:
        # Attempt Cholesky decomposition
        np.linalg.cholesky(Sigma.values)
    except np.linalg.LinAlgError:
        log.warning("Covariance matrix is not positive definite. Applying regularization.")
        min_eig = np.min(np.linalg.eigvalsh(Sigma.values))
        Sigma += np.eye(len(syms)) * (max(0, -min_eig) + 1e-8)

    w_prev = _load_previous_weights(syms).reindex(syms).fillna(0.0)
    nav = STARTING_CAPITAL
    # Liquidity cap: maximum percentage of ADV per name, clipped by per-name cap
    liq_cap = (LIQUIDITY_MAX_PCT_ADV * adv / nav).fillna(MAX_POSITION_WEIGHT)
    # Combine liquidity cap and global max position cap by taking the element-wise minimum
    cap = np.minimum(liq_cap.values, MAX_POSITION_WEIGHT)

    n = len(syms)
    w = cp.Variable(n)

    mu = a.values
    S = Sigma.values
    # Transaction cost term penalizes turnover
    tcost = MVO_COST_LAMBDA * cp.norm1(w - w_prev.values)
    # Objective uses dynamic risk lambda
    obj = cp.Maximize(mu @ w - effective_lambda * cp.quad_form(w, S) - tcost)

    cons = []
    # Gross leverage constraint
    cons.append(cp.sum(cp.abs(w)) <= GROSS_LEVERAGE + 1e-6)
    # Net exposure constraint
    cons.append(cp.sum(w) == NET_EXPOSURE)
    # Long-only and per-name caps
    cons.append(w >= 0.0)
    cons.append(w <= cap)
    # Beta constraints
    try:
        betas = est_beta_asof(syms, as_of, market_symbol="IWM", lookback=63)
        betas = betas.reindex(syms).fillna(0.0).values
        cons.append(betas @ w >= BETA_MIN)
        cons.append(betas @ w <= BETA_MAX)
        log.info(f"Added beta constraints: [{BETA_MIN}, {BETA_MAX}]")
    except Exception as e:
        log.warning(f"Could not add beta constraints: {e}")
    # Turnover constraint (annualized limit converted to per period)
    step_turnover = TURNOVER_LIMIT_ANNUAL / 252.0
    cons.append(cp.norm1(w - w_prev.values) <= step_turnover + 1e-6)

    prob = cp.Problem(obj, cons)
    try:
        log.info("Solving MVO optimization...")
        prob.solve(solver=cp.ECOS, warm_start=True, max_iters=10000, verbose=False)
        if prob.status not in ['optimal', 'optimal_inaccurate']:
            log.warning(f"Optimization status: {prob.status}")
            return _fallback_optimizer(a)
        if w.value is None:
            log.warning("Optimization returned None")
            return _fallback_optimizer(a)
        sol_values = np.array(w.value).flatten()
        sol = pd.Series(sol_values, index=syms)
        sol = sol[sol.abs() > 1e-6]
        log.info(f"MVO optimization successful: {len(sol)} positions, gross: {sol.abs().sum():.3f}, net: {sol.sum():.3f}")
        return sol.sort_values(ascending=False)
    except Exception as e:
        log.error(f"Optimization failed: {e}")
        return _fallback_optimizer(a)

def _fallback_optimizer(alpha: pd.Series) -> pd.Series:
    a = alpha.dropna().sort_values(ascending=False).clip(lower=0.0)
    if a.empty:
        return pd.Series(dtype=float)
    n = max(1, min(20, len(a)))
    top_alpha = a.head(n)
    target_weight = GROSS_LEVERAGE / n
    w = pd.Series(target_weight, index=top_alpha.index)
    log.info(f"Fallback optimization: {len(w)} equal-weight positions")
    return w

def estimate_portfolio_risk(weights: pd.Series, as_of: date) -> dict:
    if weights.empty:
        return {}
    try:
        syms = weights.index.tolist()
        Sigma, factor_exposures = synthesize_covariance(syms, as_of)
        if Sigma.empty:
            return {'portfolio_vol': 0.0, 'error': 'No risk model available'}
        w = weights.reindex(Sigma.index).fillna(0.0).values
        portfolio_var = w.T @ Sigma.values @ w
        portfolio_vol = np.sqrt(portfolio_var * 252)
        try:
            from risk.risk_model import est_beta_asof  # type: ignore
            betas = est_beta_asof(syms, as_of, market_symbol="IWM", lookback=63)
            portfolio_beta = (weights.reindex(betas.index).fillna(0.0) * betas).sum()
        except Exception:
            portfolio_beta = 0.0
        factor_risk = {}
        if not factor_exposures.empty:
            portfolio_exposures = factor_exposures.T @ weights.reindex(factor_exposures.index).fillna(0.0)
            factor_risk = portfolio_exposures.to_dict()
        return {
            'portfolio_vol': portfolio_vol,
            'portfolio_beta': portfolio_beta,
            'gross_leverage': weights.abs().sum(),
            'net_exposure': weights.sum(),
            'factor_exposures': factor_risk,
            'n_positions': len(weights[weights.abs() > 1e-6])
        }
    except Exception as e:
        log.error(f"Error estimating portfolio risk: {e}")
        return {'error': str(e)}
