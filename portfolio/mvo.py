"""
Mean–variance portfolio optimizer.

This module implements a simple mean–variance optimization (MVO) routine
that constructs portfolio weights based on expected returns (alphas) and
estimated covariance of asset returns.  It solves for weights that
maximize risk‑adjusted return under a gross leverage constraint and
enforces long‑only weights by clipping negatives.  The routine falls
back to equal weighting when covariance estimation fails.

It is designed to be used by ``portfolio.optimizer.build_portfolio`` when
the configuration flag ``USE_MVO`` is set.  The expected returns are
provided via the ``alpha`` series, and historical prices are loaded
from the ``daily_bars`` table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import text, bindparam
from db import engine
from utils.price_utils import price_expr
from config import GROSS_LEVERAGE, MAX_POSITION_WEIGHT
import logging

log = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import cvxpy as cp  # type: ignore
except Exception:  # pragma: no cover - CVXPY is optional
    cp = None


def _fallback_optimizer(alpha: pd.Series) -> pd.Series:
    """Construct a simple long-only portfolio using alpha magnitudes."""

    if alpha.empty:
        return pd.Series(dtype=float)

    weights = alpha.clip(lower=0.0)
    if weights.sum() == 0:
        weights = pd.Series(1.0, index=alpha.index)

    weights = weights / weights.sum() * GROSS_LEVERAGE

    if MAX_POSITION_WEIGHT > 0:
        weights = np.minimum(weights, MAX_POSITION_WEIGHT)
        gross = weights.sum()
        if gross > 0:
            weights = weights * (GROSS_LEVERAGE / gross)

    return weights


def _latest_prices(symbols: list[str]) -> pd.Series:
    """Placeholder for latest price lookup; patched in tests."""

    raise RuntimeError("Price lookup not configured in test environment")


def _adv20(symbols: list[str]) -> pd.Series:
    """Placeholder for ADV20 lookup; patched in tests."""

    raise RuntimeError("ADV lookup not configured in test environment")


def synthesize_covariance(symbols: list[str], lookback: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Placeholder covariance synthesis; patched in tests."""

    raise RuntimeError("Covariance synthesis not configured in test environment")


def _load_return_history(symbols: list[str], end_date: date, lookback: int = 252) -> pd.DataFrame:
    """Load daily percentage returns for the given symbols.

    Parameters
    ----------
    symbols : list[str]
        Symbols to load returns for.
    end_date : date
        End date (inclusive) of the lookback window.
    lookback : int
        Number of trading days to include.

    Returns
    -------
    pd.DataFrame
        DataFrame with index as timestamp and columns as symbols.  Values
        are daily percentage returns.
    """
    if not symbols:
        return pd.DataFrame()
    start_date = end_date - timedelta(days=lookback * 2)  # buffer to ensure sufficient rows
    stmt = text(f"""
        SELECT symbol, ts, {price_expr()} AS px
        FROM daily_bars
        WHERE symbol IN :syms AND ts BETWEEN :start AND :end
        ORDER BY ts
    """).bindparams(bindparam("syms", expanding=True))
    df = pd.read_sql_query(stmt, engine, params={"syms": tuple(symbols), "start": start_date, "end": end_date}, parse_dates=["ts"])
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(["symbol", "ts"]).copy()
    # Compute percentage returns per symbol
    df["ret"] = df.groupby("symbol")["px"].pct_change()
    pivot = df.pivot_table(index="ts", columns="symbol", values="ret")
    # Take the most recent lookback rows
    pivot = pivot.dropna(how="all").tail(lookback)
    return pivot


def build_portfolio_mvo(alpha: pd.Series, as_of: date, lookback: int = 252, shrinkage: float = 1e-6) -> pd.Series:
    """Construct a long‑only mean–variance optimized portfolio.

    Parameters
    ----------
    alpha : pd.Series
        Series indexed by symbol containing expected returns or alphas.
    as_of : date
        Date for which to construct the portfolio.
    lookback : int
        Number of trading days to use for covariance estimation.
    shrinkage : float
        Shrinkage parameter added to the diagonal of the covariance matrix
        to improve numerical stability.

    Returns
    -------
    pd.Series
        Portfolio weights indexed by symbol.  Weights sum to the target
        gross leverage defined in ``config.GROSS_LEVERAGE``.  Negative
        weights are clipped to zero.
    """
    symbols = list(alpha.index)
    if not symbols:
        return pd.Series(dtype=float)
    # Load historical returns
    returns = _load_return_history(symbols, as_of, lookback)
    # Align alpha with available symbols
    symbols = [s for s in symbols if s in returns.columns]
    if not symbols:
        log.warning("MVO: No return history available; using fallback optimizer.")
        return _fallback_optimizer(alpha)
    returns = returns[symbols].fillna(0.0)
    # Compute covariance matrix and shrinkage
    Sigma = returns.cov().values
    n = Sigma.shape[0]
    Sigma += np.eye(n) * shrinkage
    # Use provided alpha as expected returns; align ordering
    mu = alpha.loc[symbols].values
    # Solve unconstrained mean‑variance weights: w ∝ Σ⁻¹ μ
    try:
        inv = np.linalg.pinv(Sigma)
        w_raw = inv @ mu
        # Clip negative weights to zero (long‑only)
        w_raw = np.clip(w_raw, 0.0, None)
        if w_raw.sum() == 0:
            # All weights clipped; revert to equal weighting
            w_raw = np.ones_like(w_raw)
        # Normalise to target gross leverage
        w = w_raw / w_raw.sum() * GROSS_LEVERAGE
        # Enforce maximum position cap
        max_cap = MAX_POSITION_WEIGHT
        if max_cap > 0:
            w = np.minimum(w, max_cap)
            # re‑scale after capping
            gross = w.sum()
            if gross > 0:
                w = w * (GROSS_LEVERAGE / gross)
        return pd.Series(w, index=symbols)
    except Exception as e:
        log.error(f"MVO optimization failed: {e}")
        return _fallback_optimizer(alpha.reindex(symbols, fill_value=0.0))