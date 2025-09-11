
from __future__ import annotations

import logging
from datetime import date
import numpy as np
import pandas as pd
from sqlalchemy import text, bindparam
from db import engine  # type: ignore
from utils.price_utils import select_price_as

try:
    from config import (
        LONG_COUNT_MIN, LONG_COUNT_MAX, MAX_PER_SECTOR,
        GROSS_LEVERAGE, NET_EXPOSURE, MAX_POSITION_WEIGHT,
        MIN_PRICE, MIN_ADV_USD, BETA_HEDGE_SYMBOL, BETA_HEDGE_MAX_WEIGHT, BETA_TARGET,
        MAX_NAME_CORR, SECTOR_NEUTRALIZE, USE_QP_OPTIMIZER, QP_CORR_PENALTY,
        USE_MVO
    )
except Exception:
    LONG_COUNT_MIN, LONG_COUNT_MAX, MAX_PER_SECTOR = 10, 30, 5
    GROSS_LEVERAGE, NET_EXPOSURE, MAX_POSITION_WEIGHT = 1.0, 1.0, 0.10
    MIN_PRICE, MIN_ADV_USD = 1.0, 100000.0
    BETA_HEDGE_SYMBOL, BETA_HEDGE_MAX_WEIGHT, BETA_TARGET = "IWM", 0.20, 1.0
    MAX_NAME_CORR, SECTOR_NEUTRALIZE = 0.85, True
    USE_QP_OPTIMIZER, QP_CORR_PENALTY, USE_MVO = True, 0.5, False

log = logging.getLogger(__name__)

def _latest_prices(symbols: list[str]) -> pd.Series:
    if not symbols: return pd.Series(dtype=float)
    stmt = text(f"""
        WITH latest AS (SELECT symbol, MAX(ts) ts FROM daily_bars WHERE symbol IN :syms GROUP BY symbol)
        SELECT b.symbol, {select_price_as('px')}
        FROM daily_bars b JOIN latest l ON b.symbol=l.symbol AND b.ts=l.ts
    """).bindparams(bindparam("syms", expanding=True))
    df = pd.read_sql_query(stmt, engine, params={'syms': tuple(symbols)})
    return df.set_index("symbol")["px"] if not df.empty else pd.Series(dtype=float)

def _adv20(symbols: list[str]) -> pd.Series:
    if not symbols: return pd.Series(dtype=float)
    stmt = text("SELECT symbol, adv_usd_20 FROM universe WHERE symbol IN :syms").bindparams(bindparam("syms", expanding=True))
    df = pd.read_sql_query(stmt, engine, params={'syms': tuple(symbols)})
    return df.set_index("symbol")["adv_usd_20"] if not df.empty else pd.Series(dtype=float)

def _pairwise_corr_filter(cands: pd.DataFrame, max_corr: float = 0.85) -> list[str]:
    if cands.empty: return []
    cols = [c for c in ['size_ln','mom_21','turnover_21','beta_63'] if c in cands.columns]
    X = cands[cols].fillna(0.0).values
    keep: list[str] = []
    for i, row in enumerate(X):
        s = cands.iloc[i]['symbol']
        if not keep:
            keep.append(s); continue
        Y = cands[cands['symbol'].isin(keep)][cols].fillna(0.0).values
        denom = (np.linalg.norm(row)+1e-12) * (np.linalg.norm(Y, axis=1)+1e-12)
        sim = (Y @ row) / denom
        if np.all(sim < max_corr): keep.append(s)
    return keep

def build_portfolio(pred_df: pd.DataFrame, as_of: date, current_symbols: list[str] | None = None) -> pd.Series:
    if pred_df is None or pred_df.empty:
        return pd.Series(dtype=float)

    if USE_MVO:
        try:
            from portfolio.mvo import build_portfolio_mvo
            alpha = pred_df.set_index('symbol')['y_pred']
            return build_portfolio_mvo(alpha, as_of)
        except Exception as e:
            log.warning(f"MVO optimization failed, falling back to heuristic: {e}")

    pred_df = pred_df.copy().dropna(subset=['y_pred']).sort_values('y_pred', ascending=False)
    px = _latest_prices(pred_df['symbol'].tolist())
    adv = _adv20(pred_df['symbol'].tolist())
    pred_df['px'] = pred_df['symbol'].map(px); pred_df['adv20'] = pred_df['symbol'].map(adv)
    pred_df = pred_df[(pred_df['px'].fillna(0)>=MIN_PRICE) & (pred_df['adv20'].fillna(0)>=MIN_ADV_USD)]

    if not pred_df.empty:
        pred_df['score_z'] = (pred_df['y_pred'] - pred_df['y_pred'].mean()) / (pred_df['y_pred'].std(ddof=0) or 1.0)
        unique_syms = _pairwise_corr_filter(pred_df[['symbol','score_z','size_ln','mom_21','turnover_21','beta_63']].copy(), MAX_NAME_CORR)
        if unique_syms:
            pred_df = pred_df[pred_df['symbol'].isin(unique_syms)]

    try:
        from risk.sector import sector_asof  # type: ignore
        sec = sector_asof(pred_df['symbol'].tolist(), as_of)
    except Exception:
        sec = {s: 'UNKNOWN' for s in pred_df['symbol'].tolist()}

    long_bucket, counts = [], {}
    for _, r in pred_df.iterrows():
        s = r['symbol']; sector = str(sec.get(s, 'UNKNOWN'))
        if r['y_pred'] <= 0: break
        if counts.get(sector,0) < MAX_PER_SECTOR:
            long_bucket.append(s); counts[sector] = counts.get(sector,0)+1
        if len(long_bucket) >= LONG_COUNT_MAX: break

    nL = max(LONG_COUNT_MIN, min(LONG_COUNT_MAX, len(long_bucket)))
    long_syms = long_bucket[:nL]

    L = max(0.0, (GROSS_LEVERAGE + NET_EXPOSURE) / 2.0)
    per_w_L = min(L / max(1,len(long_syms)), MAX_POSITION_WEIGHT)

    if len(long_syms) >= 5:
        try:
            from portfolio.qp_optimizer import solve_qp
            cols = [c for c in ['size_ln','mom_21','turnover_21','beta_63'] if c in pred_df.columns]
            Z = pred_df[pred_df['symbol'].isin(long_syms)][['symbol']+cols].set_index('symbol').fillna(0.0)
            X = Z.values
            norm = (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            Xn = X / norm
            C = (Xn @ Xn.T) * 0.5  # QP_CORR_PENALTY default-ish
            mu = pred_df.set_index('symbol').loc[long_syms, 'y_pred']
            qp_w = solve_qp(mu, C, gross=L, w_cap=per_w_L)
            if qp_w is not None:
                w = qp_w.clip(lower=0.0)
                s = float(w.abs().sum()) or 1.0
                w = w * (L / s)
                weights = w
            else:
                weights = pd.Series({s: per_w_L for s in long_syms})
        except Exception:
            weights = pd.Series({s: per_w_L for s in long_syms})
    else:
        weights = pd.Series({s: per_w_L for s in long_syms})

    try:
        from risk.risk_model import portfolio_beta  # type: ignore
        b = portfolio_beta(weights, as_of, BETA_HEDGE_SYMBOL, lookback=63)
        hedge_w = - (b - BETA_TARGET)
        hedge_w = max(min(hedge_w, BETA_HEDGE_MAX_WEIGHT), -BETA_HEDGE_MAX_WEIGHT)
        if abs(hedge_w) > 1e-4:
            weights[BETA_HEDGE_SYMBOL] = weights.get(BETA_HEDGE_SYMBOL, 0.0) + hedge_w
    except Exception:
        pass

    gross = weights.abs().sum()
    tgt_gross = L + (weights[weights<0].abs().sum() if (weights<0).any() else 0.0)
    if gross > 0:
        weights *= (tgt_gross / gross)
    return weights.sort_values(ascending=False)
