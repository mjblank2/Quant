from __future__ import annotations
from datetime import date
import logging
import numpy as np
import pandas as pd
from sqlalchemy import text, bindparam
from db import engine
from utils.price_utils import select_price_as
from config import (
    LONG_COUNT_MIN, LONG_COUNT_MAX, MAX_PER_SECTOR,
    GROSS_LEVERAGE, NET_EXPOSURE, MAX_POSITION_WEIGHT,
    MIN_PRICE, MIN_ADV_USD, BETA_HEDGE_SYMBOL, BETA_HEDGE_MAX_WEIGHT, BETA_TARGET,
    MAX_NAME_CORR, USE_QP_OPTIMIZER, QP_CORR_PENALTY, USE_MVO
)
from risk.sector import sector_asof
from risk.risk_model import portfolio_beta

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
    keep = []
    for i, row in enumerate(X):
        s = cands.iloc[i]['symbol']
        if not keep:
            keep.append(s); continue
        Y = cands[cands['symbol'].isin(keep)][cols].fillna(0.0).values
        denom = (np.linalg.norm(row)+1e-12) * (np.linalg.norm(Y, axis=1)+1e-12)
        sim = (Y @ row) / denom
        if np.all(sim < max_corr): keep.append(s)
    return keep

def build_portfolio(pred_df: pd.DataFrame, as_of: date) -> pd.Series:
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

    sec = sector_asof(pred_df['symbol'].tolist(), as_of)
    long_bucket, counts = [], {}
    for _, r in pred_df.iterrows():
        s = r['symbol']; sector = str(sec.get(s, 'UNKNOWN')) if hasattr(sec,'get') else (sec.loc[s] if s in sec.index else 'UNKNOWN')
        if r['y_pred'] <= 0: break
        if counts.get(sector,0) < MAX_PER_SECTOR:
            long_bucket.append(s); counts[sector] = counts.get(sector,0)+1
        if len(long_bucket) >= LONG_COUNT_MAX: break

    nL = max(LONG_COUNT_MIN, min(LONG_COUNT_MAX, len(long_bucket)))
    long_syms = long_bucket[:nL]
    L = max(0.0, (GROSS_LEVERAGE + NET_EXPOSURE) / 2.0)
    per_w_L = min(L / max(1,len(long_syms)), MAX_POSITION_WEIGHT)
    weights = pd.Series({s: per_w_L for s in long_syms})

    try:
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
