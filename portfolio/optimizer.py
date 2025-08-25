from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date
from sqlalchemy import text, bindparam
from db import engine
from risk.sector import sector_asof, build_sector_dummies
from risk.risk_model import portfolio_beta, est_beta_asof
from config import (
    USE_QP_OPTIMIZER, QP_CORR_PENALTY,
    LONG_COUNT_MIN, LONG_COUNT_MAX, SHORT_COUNT_MAX, MAX_PER_SECTOR,
    GROSS_LEVERAGE, NET_EXPOSURE, MAX_POSITION_WEIGHT,
    MIN_PRICE, MIN_ADV_USD, BETA_HEDGE_SYMBOL, BETA_HEDGE_MAX_WEIGHT, BETA_TARGET,
    MAX_NAME_CORR, SECTOR_NEUTRALIZE
)
from tax.lots import tax_sell_penalty_bps

def _latest_prices(symbols: list[str]) -> pd.Series:
    if not symbols: return pd.Series(dtype=float)
    stmt = text("""
        WITH latest AS (SELECT symbol, MAX(ts) ts FROM daily_bars WHERE symbol IN :syms GROUP BY symbol)
        SELECT b.symbol, COALESCE(b.adj_close, b.close) AS px
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
    # cands: symbol, score (must be z-scored), fallback to uniqueness-only filter
    if cands.empty:
        return []
    # Build rolling cosine-similarity proxy using factor vector (size, mom_21, turnover_21, beta_63) if present
    cols = [c for c in ['size_ln','mom_21','turnover_21','beta_63'] if c in cands.columns]
    X = cands[cols].fillna(0.0).values
    keep = []
    for i, row in enumerate(X):
        s = cands.iloc[i]['symbol']
        if not keep:
            keep.append(s); continue
        Y = cands[cands['symbol'].isin(keep)][cols].fillna(0.0).values
        # cosine similarity
        denom = (np.linalg.norm(row)+1e-12) * (np.linalg.norm(Y, axis=1)+1e-12)
        sim = (Y @ row) / denom
        if np.all(sim < max_corr):
            keep.append(s)
    return keep

def _exposure_limits(symbols: list[str], as_of) -> pd.Series:
    # Build sector caps
    sec = sector_asof(symbols, as_of).fillna("UNKNOWN")
    counts = {}
    cap = pd.Series(index=symbols, dtype=int)
    for s in symbols:
        key = str(sec.get(s) if hasattr(sec,'get') else (sec.loc[s] if s in sec.index else "UNKNOWN"))
        counts[key] = counts.get(key, 0)
        cap[s] = counts[key]
    return sec

from .qp_optimizer import optimize_qp

def build_portfolio(pred_df: pd.DataFrame, as_of: date, current_symbols: list[str] | None = None) -> pd.Series:
    """
    Inputs:
      pred_df columns: symbol, y_pred, vol_21, adv_usd_21, size_ln, mom_21, turnover_21, beta_63
    Returns weights per symbol (positive=long, negative=short); may include BETA_HEDGE_SYMBOL if hedge applied.
    """
    if pred_df is None or pred_df.empty:
        return pd.Series(dtype=float)

    pred_df = pred_df.copy().sort_values("y_pred", ascending=False)
    pred_df = pred_df.dropna(subset=["y_pred"])
    # Price/ADV gates
    px = _latest_prices(pred_df["symbol"].tolist())
    adv = _adv20(pred_df["symbol"].tolist())
    pred_df["px"] = pred_df["symbol"].map(px); pred_df["adv20"] = pred_df["symbol"].map(adv)
    pred_df = pred_df[(pred_df["px"].fillna(0) >= MIN_PRICE) & (pred_df["adv20"].fillna(0) >= MIN_ADV_USD)]

    # De-crowding: favor uniqueness by filtering highly factor-similar picks first
    if not pred_df.empty:
        pred_df["score_z"] = (pred_df["y_pred"] - pred_df["y_pred"].mean())/ (pred_df["y_pred"].std(ddof=0) or 1.0)
        unique_syms = _pairwise_corr_filter(pred_df[["symbol","score_z","size_ln","mom_21","turnover_21","beta_63"]].copy(), MAX_NAME_CORR)
        if unique_syms:
            pred_df = pred_df[pred_df["symbol"].isin(unique_syms)]

    # Sector balanced selection
    sec = sector_asof(pred_df["symbol"].tolist(), as_of)
    max_per_sec = MAX_PER_SECTOR
    long_bucket = []
    counts = {}
    for _, r in pred_df.iterrows():
        s = r["symbol"]; sector = str(sec.get(s, "UNKNOWN")) if hasattr(sec,'get') else (sec.loc[s] if s in sec.index else "UNKNOWN")
        if r["y_pred"] <= 0: break
        if counts.get(sector, 0) < max_per_sec:
            long_bucket.append(s); counts[sector] = counts.get(sector,0) + 1
        if len(long_bucket) >= LONG_COUNT_MAX: break

    # Respect cardinality
    nL = max(LONG_COUNT_MIN, min(LONG_COUNT_MAX, len(long_bucket)))
    long_syms = long_bucket[:nL]

    L = max(0.0, (GROSS_LEVERAGE + NET_EXPOSURE) / 2.0)
    if USE_QP_OPTIMIZER and len(long_syms) > 1:
        # QP on the candidate subset
        cand = pred_df[pred_df['symbol'].isin(long_syms)][['symbol','y_pred','size_ln','mom_21','turnover_21','beta_63']].copy()
        w_ser = optimize_qp(cand, long_budget=L, max_per_name=MAX_POSITION_WEIGHT, corr_penalty=QP_CORR_PENALTY)
        weights = w_ser
    else:
        per_w_L = min(L / max(1, len(long_syms)), MAX_POSITION_WEIGHT)
        weights = pd.Series({s: per_w_L for s in long_syms}, dtype=float)

    # Tax-aware *sell* penalty: if current_symbols provided, discourage dropping names that trigger ST gains
    if current_symbols:
        cur_set = set(current_symbols)
        to_drop = [s for s in cur_set if s not in long_syms]
        # apply penalty by soft-keeping expensive sells: if penalty large, keep minimal placeholder weight
        for s in to_drop:
            pen_bps = tax_sell_penalty_bps(s, as_of)
            if pen_bps > 0:
                # Keep a micro-weight to avoid immediate realization; user can manually override
                weights[s] = min(per_w_L*0.25, MAX_POSITION_WEIGHT*0.25)

    # Optional: market-beta hedge
    try:
        b = portfolio_beta(weights, as_of, BETA_HEDGE_SYMBOL, lookback=63)
        hedge_w = - (b - BETA_TARGET)  # unit beta hedge weight (approx)
        hedge_w = max(min(hedge_w, BETA_HEDGE_MAX_WEIGHT), -BETA_HEDGE_MAX_WEIGHT)
        if abs(hedge_w) > 1e-4:
            weights[BETA_HEDGE_SYMBOL] = weights.get(BETA_HEDGE_SYMBOL, 0.0) + hedge_w
    except Exception:
        pass

    # Re-standardize to gross leverage budget
    gross = weights.abs().sum()
    tgt_gross = L + (weights[weights<0].abs().sum() if (weights<0).any() else 0.0)
    if gross > 0:
        weights *= (tgt_gross / gross)
    return weights.sort_values(ascending=False)

