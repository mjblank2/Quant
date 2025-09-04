
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

def _latest_prices(symbols: list[str]) -> pd.Series:
    if not symbols:
        return pd.Series(dtype=float)
    stmt = text("""
        WITH latest AS (
            SELECT symbol, MAX(ts) ts
            FROM daily_bars
            WHERE symbol = ANY(:syms)
            GROUP BY symbol
        )
        SELECT b.symbol, COALESCE(b.adj_close, b.close) AS px
        FROM daily_bars b
        JOIN latest l ON b.symbol=l.symbol AND b.ts=l.ts
    """).bindparams(bindparam("syms", expanding=True))
    try:
        df = pd.read_sql_query(stmt, engine, params={'syms': symbols})
        return df.set_index("symbol")["px"] if not df.empty else pd.Series(dtype=float)
    except Exception as e:
        log.error(f"Error fetching prices: {e}")
        return pd.Series(dtype=float)

def _adv20(symbols: list[str]) -> pd.Series:
    if not symbols:
        return pd.Series(dtype=float)
    stmt = text("SELECT symbol, adv_usd_20 FROM universe WHERE symbol = ANY(:syms)")
    try:
        df = pd.read_sql_query(stmt, engine, params={'syms': symbols})
        return df.set_index("symbol")["adv_usd_20"] if not df.empty else pd.Series(dtype=float)
    except Exception as e:
        log.error(f"Error fetching ADV: {e}")
        return pd.Series(dtype=float)

def _load_previous_weights(symbols: list[str]) -> pd.Series:
    if not symbols:
        return pd.Series(dtype=float)
    stmt = text("""
        SELECT symbol, weight
        FROM target_positions
        WHERE ts=(SELECT MAX(ts) FROM target_positions)
        AND symbol = ANY(:syms)
    """).bindparams(bindparam("syms", expanding=True))
    try:
        df = pd.read_sql_query(stmt, engine, params={'syms': symbols})
        return df.set_index('symbol')['weight'] if not df.empty else pd.Series(dtype=float)
    except Exception as e:
        log.warning(f"Error loading previous weights: {e}")
        return pd.Series(dtype=float)

def build_portfolio_mvo(alpha: pd.Series, as_of: date, risk_lambda: float | None = None) -> pd.Series:
    if cp is None:
        log.warning("CVXPY not available, falling back to simple optimization")
        return _fallback_optimizer(alpha)

    risk_lambda = risk_lambda if risk_lambda is not None else MVO_RISK_LAMBDA

    syms = alpha.index.tolist()
    px = _latest_prices(syms)
    adv = _adv20(syms)

    ok = (px >= MIN_PRICE) & (adv >= MIN_ADV_USD)
    a = alpha[ok].fillna(0.0)

    if a.empty:
        log.warning("No symbols passed filters")
        return pd.Series(dtype=float)

    syms = a.index.tolist()
    px = px.reindex(syms)
    adv = adv.reindex(syms)

    log.info(f"Building risk model for {len(syms)} symbols")
    Sigma, factor_exposures = synthesize_covariance(syms, as_of)
    if Sigma.empty:
        log.warning("Failed to build risk model, using diagonal covariance")
        Sigma = pd.DataFrame(np.eye(len(syms)) * 1e-4, index=syms, columns=syms)

    w_prev = _load_previous_weights(syms).reindex(syms).fillna(0.0)
    nav = STARTING_CAPITAL
    liq_cap = (LIQUIDITY_MAX_PCT_ADV * adv / nav).clip(upper=MAX_POSITION_WEIGHT).fillna(MAX_POSITION_WEIGHT)

    n = len(syms)
    w = cp.Variable(n)

    mu = a.values
    S = Sigma.values
    tcost = MVO_COST_LAMBDA * cp.norm1(w - w_prev.values)
    obj = cp.Maximize(mu @ w - risk_lambda * cp.quad_form(w, S) - tcost)

    cons = []
    cons += [cp.sum(cp.abs(w)) <= GROSS_LEVERAGE + 1e-6]
    cons += [cp.sum(w) == NET_EXPOSURE]
    cons += [w >= 0.0, w <= liq_cap.values]
    cons += [w <= MAX_POSITION_WEIGHT]

    try:
        betas = est_beta_asof(syms, as_of, market_symbol="IWM", lookback=63)
        betas = betas.reindex(syms).fillna(0.0).values
        cons += [betas @ w >= BETA_MIN, betas @ w <= BETA_MAX]
        log.info(f"Added beta constraints: [{BETA_MIN}, {BETA_MAX}]")
    except Exception as e:
        log.warning(f"Could not add beta constraints: {e}")

    step_turnover = TURNOVER_LIMIT_ANNUAL / 252.0
    cons += [cp.norm1(w - w_prev.values) <= step_turnover + 1e-6]

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
        sol = pd.Series(np.array(w.value).flatten(), index=syms)
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
