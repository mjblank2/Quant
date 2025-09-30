from __future__ import annotations
import numpy as np, pandas as pd
from datetime import date
from sqlalchemy import text, bindparam
from db import engine
from risk.sector import sector_asof
from risk.risk_model import portfolio_beta
from utils.price_utils import select_price_as
import logging
import os
from config import (
    LONG_COUNT_MIN, LONG_COUNT_MAX, MAX_PER_SECTOR,
    GROSS_LEVERAGE, NET_EXPOSURE, MAX_POSITION_WEIGHT,
    MIN_PRICE, MIN_ADV_USD, BETA_HEDGE_SYMBOL, BETA_HEDGE_MAX_WEIGHT, BETA_TARGET,
    MAX_NAME_CORR, SECTOR_NEUTRALIZE, USE_QP_OPTIMIZER, QP_CORR_PENALTY,
    USE_MVO
)

# Optional risk-parity settings driven by environment variables.
# If USE_RISK_PARITY is set to 'true' (case-insensitive) then the
# portfolio construction will employ a risk parity weighting scheme rather
# than simple volatility weighting or quadratic programming.  The
# TARGET_VOL environment variable controls the annualised volatility
# objective used when computing the risk parity portfolio.  If omitted,
# a default of 0.10 (10% vol) is used.  These settings can be
# configured in Render via environment variables.
USE_RISK_PARITY = os.getenv('USE_RISK_PARITY', 'false').lower() == 'true'
try:
    TARGET_VOL = float(os.getenv('TARGET_VOL', '0.10'))
except Exception:
    TARGET_VOL = 0.10

log = logging.getLogger(__name__)


def _latest_prices(symbols: list[str]) -> pd.Series:
    if not symbols:
        return pd.Series(dtype=float)
    stmt = text(f"""
        WITH latest AS (SELECT symbol, MAX(ts) ts FROM daily_bars WHERE symbol IN :syms GROUP BY symbol)
        SELECT b.symbol, {select_price_as('px')}
        FROM daily_bars b JOIN latest l ON b.symbol=l.symbol AND b.ts=l.ts
    """).bindparams(bindparam("syms", expanding=True))
    df = pd.read_sql_query(stmt, engine, params={'syms': tuple(symbols)})
    return df.set_index("symbol")["px"] if not df.empty else pd.Series(dtype=float)


def _adv20(symbols: list[str]) -> pd.Series:
    if not symbols:
        return pd.Series(dtype=float)
    stmt = text("SELECT symbol, adv_usd_20 FROM universe WHERE symbol IN :syms").bindparams(bindparam("syms", expanding=True))
    df = pd.read_sql_query(stmt, engine, params={'syms': tuple(symbols)})
    return df.set_index("symbol")["adv_usd_20"] if not df.empty else pd.Series(dtype=float)


def _pairwise_corr_filter(cands: pd.DataFrame, max_corr: float = 0.85) -> list[str]:
    if cands.empty:
        return []
    cols = [c for c in ['size_ln', 'mom_21', 'turnover_21', 'beta_63'] if c in cands.columns]
    X = cands[cols].fillna(0.0).values
    keep: list[str] = []
    for i, row in enumerate(X):
        s = cands.iloc[i]['symbol']
        if not keep:
            keep.append(s)
            continue
        Y = cands[cands['symbol'].isin(keep)][cols].fillna(0.0).values
        denom = (np.linalg.norm(row) + 1e-12) * (np.linalg.norm(Y, axis=1) + 1e-12)
        sim = (Y @ row) / denom
        if np.all(sim < max_corr):
            keep.append(s)
    return keep


def build_portfolio(pred_df: pd.DataFrame, as_of: date, current_symbols: list[str] | None = None) -> pd.Series:
    """Build a long-only portfolio from the provided prediction DataFrame.

    The heuristic optimizer selects the top-ranked names subject to basic universe
    filters (price/ADV) and pairwise correlation filtering.  It then sizes positions
    using volatility-adjusted weights.  A beta hedge is applied to achieve the
    desired beta exposure.  If quadratic programming (QP) sizing is enabled and
    conditions are met, a QP optimizer is used instead of simple volatility weighting.
    """
    # If MVO is enabled, delegate to convex optimizer for sizing; otherwise use heuristic long-bucket.
    if pred_df is None or pred_df.empty:
        return pd.Series(dtype=float)
    if USE_MVO:
        try:
            from portfolio.mvo import build_portfolio_mvo
            alpha = pred_df.set_index('symbol')['y_pred']
            return build_portfolio_mvo(alpha, as_of)
        except Exception as e:
            log.warning(f"MVO optimization failed, falling back to heuristic: {e}")
    # Original heuristic optimizer
    pred_df = pred_df.copy().dropna(subset=['y_pred']).sort_values('y_pred', ascending=False)
    px = _latest_prices(pred_df['symbol'].tolist())
    adv = _adv20(pred_df['symbol'].tolist())
    pred_df['px'] = pred_df['symbol'].map(px)
    pred_df['adv20'] = pred_df['symbol'].map(adv)
    pred_df = pred_df[(pred_df['px'].fillna(0) >= MIN_PRICE) & (pred_df['adv20'].fillna(0) >= MIN_ADV_USD)]
    if not pred_df.empty:
        pred_df['score_z'] = (pred_df['y_pred'] - pred_df['y_pred'].mean()) / (pred_df['y_pred'].std(ddof=0) or 1.0)
        unique_syms = _pairwise_corr_filter(pred_df[['symbol', 'score_z', 'size_ln', 'mom_21', 'turnover_21', 'beta_63']].copy(), MAX_NAME_CORR)
        if unique_syms:
            pred_df = pred_df[pred_df['symbol'].isin(unique_syms)]
    sec = sector_asof(pred_df['symbol'].tolist(), as_of)
    long_bucket: list[str] = []
    counts: Dict[str, int] = {}
    for _, r in pred_df.iterrows():
        s = r['symbol']
        sector = str(sec.get(s, 'UNKNOWN')) if hasattr(sec, 'get') else (sec.loc[s] if s in sec.index else 'UNKNOWN')
        if r['y_pred'] <= 0:
            break
        if counts.get(sector, 0) < MAX_PER_SECTOR:
            long_bucket.append(s)
            counts[sector] = counts.get(sector, 0) + 1
        if len(long_bucket) >= LONG_COUNT_MAX:
            break
    # Determine final number of long positions
    nL = max(LONG_COUNT_MIN, min(LONG_COUNT_MAX, len(long_bucket)))
    long_syms = long_bucket[:nL]
    if not long_syms:
        return pd.Series(dtype=float)
    # Determine weights for long positions
    alpha = pred_df.set_index('symbol')['y_pred']
    weights = pd.Series(dtype=float)
    if USE_RISK_PARITY:
        # Risk parity weighting with dynamic volatility targeting
        try:
            from portfolio.risk_parity import build_portfolio_risk_parity
            features = pred_df.set_index('symbol')
            rp_w = build_portfolio_risk_parity(alpha.loc[long_syms], features.loc[long_syms], target_vol=TARGET_VOL)
            # Scale to gross leverage
            if rp_w is not None and not rp_w.empty:
                s = rp_w.abs().sum()
                weights = rp_w * (GROSS_LEVERAGE / s) if s > 0 else rp_w
        except Exception as e:
            log.warning(f"Risk-parity sizing failed, falling back to vol weights: {e}")
    if weights.empty:
        # Fallback: volatility-adjusted weights using vol_63 column
        vol_weights = compute_vol_weights(long_syms, pred_df.set_index('symbol'), alpha)
        weights = vol_weights * GROSS_LEVERAGE
        # Optionally apply QP sizing on selected longs to penalize factor crowding
        if USE_QP_OPTIMIZER and len(long_syms) >= 5:
            try:
                from portfolio.qp_optimizer import solve_qp
                # Build a simple factor-similarity matrix C from standardized columns
                cols = [c for c in ['size_ln', 'mom_21', 'turnover_21', 'beta_63'] if c in pred_df.columns]
                Z = pred_df[pred_df['symbol'].isin(long_syms)][['symbol'] + cols].set_index('symbol').fillna(0.0)
                # cosine similarity → positive semidefinite approximation
                X = Z.values
                norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
                Xn = X / norm
                C = (Xn @ Xn.T) * QP_CORR_PENALTY
                mu = pred_df.set_index('symbol').loc[long_syms, 'y_pred']
                qp_w = solve_qp(mu, C, gross=GROSS_LEVERAGE, w_cap=MAX_POSITION_WEIGHT)
                if qp_w is not None:
                    w = qp_w.clip(lower=0.0)  # long-only
                    # rescale to gross leverage
                    s = float(w.abs().sum()) or 1.0
                    w = w * (GROSS_LEVERAGE / s)
                    weights = w
            except Exception:
                # fall back to volatility-adjusted weights
                pass
    # Beta hedge
    try:
        b = portfolio_beta(weights, as_of, BETA_HEDGE_SYMBOL, lookback=63)
        hedge_w = - (b - BETA_TARGET)
        hedge_w = max(min(hedge_w, BETA_HEDGE_MAX_WEIGHT), -BETA_HEDGE_MAX_WEIGHT)
        if abs(hedge_w) > 1e-4:
            weights[BETA_HEDGE_SYMBOL] = weights.get(BETA_HEDGE_SYMBOL, 0.0) + hedge_w
    except Exception:
        pass
    # Rescale weights to target gross leverage (accounting for hedge)
    gross = weights.abs().sum()
    tgt_gross = GROSS_LEVERAGE + (weights[weights < 0].abs().sum() if (weights < 0).any() else 0.0)
    if gross > 0:
        weights *= (tgt_gross / gross)
    return weights.sort_values(ascending=False)


def compute_vol_weights(selected_symbols, features_df, alpha_scores):
    """Compute volatility-adjusted weights for a set of selected symbols.

    The weights are proportional to the alpha scores divided by the 63-day volatility (vol_63) and
    scaled so that their absolute values sum to one.

    Parameters
    ----------
    selected_symbols : list[str]
        Symbols selected for inclusion in the long bucket.
    features_df : pd.DataFrame
        DataFrame indexed by symbol containing at least the 'vol_63' column.
    alpha_scores : pd.Series
        Series of alpha scores indexed by symbol.

    Returns
    -------
    pd.Series
        Normalized weights indexed by symbol.
    """
    vols = features_df.loc[selected_symbols, "vol_63"].abs().replace(0, np.nan)
    base_weights = alpha_scores[selected_symbols] / vols
    base_weights = base_weights.fillna(0.0)
    gross = base_weights.abs().sum()
    return base_weights / gross if gross != 0 else base_weights