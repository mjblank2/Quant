"""
JAXopt-based mean-variance optimizer for long-only portfolios.
This module uses JAXopt BoxOSQP to solve a constrained quadratic program that
minimizes portfolio variance minus expected return weighted by lambda.
It falls back to a simple heuristic optimizer if jax/jaxopt are not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import text
from db import engine
from utils.price_utils import select_price_as
from config import GROSS_LEVERAGE, MAX_POSITION_WEIGHT
import logging

try:
    import jax.numpy as jnp
    from jaxopt import BoxOSQP
    _JAX_AVAILABLE = True
except Exception:
    _JAX_AVAILABLE = False

log = logging.getLogger(__name__)

def _load_return_history(symbols: list[str], as_of: date, lookback: int) -> pd.DataFrame:
    """
    Load historical daily returns for the provided symbols.
    Returns a DataFrame indexed by timestamp and with columns for each symbol.
    """
    if not symbols:
        return pd.DataFrame()
    start_date = as_of - timedelta(days=int(lookback * 1.5))
    end_date = as_of
    # Query returns from daily_bars table
    stmt = text("""
        SELECT symbol, ts, (px/lag_px - 1.0) AS ret
        FROM (
            SELECT b.symbol, b.ts, b.px,
                   LAG(b.px) OVER (PARTITION BY b.symbol ORDER BY b.ts) AS lag_px
            FROM daily_bars b
            WHERE b.symbol IN :syms AND b.ts BETWEEN :start AND :end
        ) sub
        WHERE lag_px IS NOT NULL
        ORDER BY ts
    """)
    df = pd.read_sql_query(stmt, engine, params={
        "syms": tuple(symbols),
        "start": start_date,
        "end": end_date
    })
    if df.empty:
        return df
    df = df.pivot_table(index='ts', columns='symbol', values='ret')
    df = df.tail(lookback)
    return df

def _solve_mvo_jaxopt(Sigma: np.ndarray, mu: np.ndarray, gross: float, w_cap: float) -> np.ndarray | None:
    """
    Solve the mean–variance optimization problem with JAXopt.
    Minimize 0.5 * w.T @ Sigma @ w - mu.T @ w
    subject to 1^T w = gross and 0 <= w <= w_cap.
    Returns None if JAXopt or jax is unavailable or fails.
    """
    if not _JAX_AVAILABLE:
        return None
    try:
        N = mu.shape[0]
        P = 0.5 * (Sigma + Sigma.T) + 1e-6 * np.eye(N)
        q = -mu
        P_j = jnp.array(P)
        q_j = jnp.array(q)
        A = jnp.ones((1, N))
        lb = jnp.array([gross])
        ub = jnp.array([gross])
        solver = BoxOSQP(
            tol=1e-6,
            maxiter=4000,
            check_primal_dual_infeasability=False,
        )
        res = solver.run(
            params_obj=(P_j, q_j, A),
            l=jnp.zeros(N),
            u=jnp.ones(N) * w_cap,
            lb=lb,
            ub=ub
        )
        w = np.array(res.params)
        return w
    except Exception as e:
        log.warning(f"JAXopt MVO failed: {e}")
        return None

def _fallback_optimizer(alpha: pd.Series) -> pd.Series:
    """
    Simple fallback optimizer: equal-weight allocation subject to gross leverage and max position weight.
    """
    symbols = list(alpha.index)
    if not symbols:
        return pd.Series(dtype=float)
    n = len(symbols)
    gross = GROSS_LEVERAGE
    w = np.ones(n) * gross / n
    max_cap = MAX_POSITION_WEIGHT if MAX_POSITION_WEIGHT > 0 else gross
    w = np.minimum(w, max_cap)
    total = w.sum()
    if total > 0:
        w = w / total * gross
    return pd.Series(w, index=symbols)

def build_portfolio_mvo(alpha: pd.Series, as_of: date, lookback: int = 252, shrinkage: float = 1e-6) -> pd.Series:
    """
    Construct a long-only mean-variance optimized portfolio.
    Uses JAXopt if available, otherwise falls back to a simple heuristic.
    """
    symbols = list(alpha.index)
    if not symbols:
        return pd.Series(dtype=float)

    returns = _load_return_history(symbols, as_of, lookback)
    symbols = [s for s in symbols if s in returns.columns]
    if not symbols:
        log.warning("MVO: No return history available; using fallback optimizer.")
        return _fallback_optimizer(alpha)

    returns = returns[symbols].fillna(0.0)
    Sigma = returns.cov().values
    Sigma += shrinkage * np.eye(len(Sigma))
    mu = alpha.loc[symbols].values

    gross = GROSS_LEVERAGE
    max_cap = MAX_POSITION_WEIGHT if MAX_POSITION_WEIGHT > 0 else gross

    w = _solve_mvo_jaxopt(Sigma, mu, gross, max_cap)
    if w is not None:
        w = np.clip(w, 0.0, None)
        total = w.sum()
        if total > 0:
            w = w / total * gross
        w = np.minimum(w, max_cap)
        total = w.sum()
        if total > 0:
            w = w / total * gross
        return pd.Series(w, index=symbols)

    log.info("Using fallback mean-variance heuristic")
    return _fallback_optimizer(alpha.reindex(symbols).fillna(0.0))
