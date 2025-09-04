from __future__ import annotations
from datetime import date
import logging
import numpy as np
import pandas as pd
try:
    import cvxpy as cp
except Exception:
    cp = None
from sqlalchemy import text, bindparam
from db import engine
from config import (
    GROSS_LEVERAGE, NET_EXPOSURE, MAX_POSITION_WEIGHT, MIN_PRICE, MIN_ADV_USD,
    MVO_RISK_LAMBDA, MVO_COST_LAMBDA, BETA_MIN, BETA_MAX, TURNOVER_LIMIT_ANNUAL,
    LIQUIDITY_MAX_PCT_ADV, STARTING_CAPITAL
)
from risk.risk_model import est_beta_asof
from risk.factor_model import synthesize_covariance

log = logging.getLogger(__name__)

def _latest_prices(symbols: list[str]) -> pd.Series:
    if not symbols: return pd.Series(dtype=float)
    stmt = text("""
        WITH latest AS (
            SELECT symbol, MAX(ts) ts
            FROM daily_bars
            WHERE symbol IN :syms
            GROUP BY symbol
        )
        SELECT b.symbol, COALESCE(b.adj_close, b.close) AS px
        FROM daily_bars b
        JOIN latest l ON b.symbol=l.symbol AND b.ts=l.ts
    """).bindparams(bindparam("syms", expanding=True))
    df = pd.read_sql_query(stmt, engine, params={'syms': tuple(symbols)})
    return df.set_index("symbol")["px"] if not df.empty else pd.Series(dtype=float)

def _adv20(symbols: list[str]) -> pd.Series:
    if not symbols: return pd.Series(dtype=float)
    stmt = text("SELECT symbol, adv_usd_20 FROM universe WHERE symbol IN :syms").bindparams(bindparam("syms", expanding=True))
    df = pd.read_sql_query(stmt, engine, params={'syms': tuple(symbols)})
    return df.set_index("symbol")["adv_usd_20"] if not df.empty else pd.Series(dtype=float)

def _load_previous_weights(symbols: list[str]) -> pd.Series:
    if not symbols: return pd.Series(dtype=float)
    stmt = text("""
        SELECT symbol, weight
        FROM target_positions
        WHERE ts=(SELECT MAX(ts) FROM target_positions)
          AND symbol IN :syms
    """).bindparams(bindparam("syms", expanding=True))
    df = pd.read_sql_query(stmt, engine, params={'syms': tuple(symbols)})
    return df.set_index('symbol')['weight'] if not df.empty else pd.Series(dtype=float)

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

    Sigma, factor_exposures = synthesize_covariance(syms, as_of)
    if Sigma.empty:
        Sigma = pd.DataFrame(np.eye(len(syms)) * 1e-4, index=syms, columns=syms)

    w_prev = _load_previous_weights(syms).reindex(syms).fillna(0.0)
    nav = STARTING_CAPITAL
    liq_cap = (LIQUIDITY_MAX_PCT_ADV * adv / nav).clip(upper=MAX_POSITION_WEIGHT).fillna(MAX_POSITION_WEIGHT)
    cap = np.minimum(liq_cap.values, MAX_POSITION_WEIGHT)

    n = len(syms)
    w = cp.Variable(n)
    mu = a.values
    S = Sigma.values
    tcost = MVO_COST_LAMBDA * cp.norm1(w - w_prev.values)
    obj = cp.Maximize(mu @ w - risk_lambda * cp.quad_form(w, S) - tcost)

    cons = [
        cp.sum(cp.abs(w)) <= GROSS_LEVERAGE + 1e-6,
        cp.sum(w) == NET_EXPOSURE,
        w >= 0.0,
        w <= cap,
    ]

    try:
        betas = est_beta_asof(syms, as_of, market_symbol="IWM", lookback=63).reindex(syms).fillna(0.0).values
        cons += [betas @ w >= BETA_MIN, betas @ w <= BETA_MAX]
    except Exception as e:
        log.warning(f"Could not add beta constraints: {e}")

    step_turnover = TURNOVER_LIMIT_ANNUAL / 252.0
    cons += [cp.norm1(w - w_prev.values) <= step_turnover + 1e-6]

    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.ECOS, warm_start=True, max_iters=10000, verbose=False)
        if prob.status not in ['optimal','optimal_inaccurate'] or w.value is None:
            log.warning(f"Optimization status: {prob.status}")
            return _fallback_optimizer(a)
        sol = pd.Series(np.array(w.value).flatten(), index=syms)
        sol = sol[sol.abs() > 1e-6]
        return sol.sort_values(ascending=False)
    except Exception as e:
        log.error(f"Optimization failed: {e}")
        return _fallback_optimizer(a)

def _fallback_optimizer(alpha: pd.Series) -> pd.Series:
    a = alpha.dropna().sort_values(ascending=False).clip(lower=0.0)
    if a.empty: return pd.Series(dtype=float)
    n = max(1, min(20, len(a)))
    per = GROSS_LEVERAGE / n
    return pd.Series(per, index=a.head(n).index)
