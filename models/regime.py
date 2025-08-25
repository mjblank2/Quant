from __future__ import annotations
import pandas as pd
import numpy as np
from sqlalchemy import text
from db import engine

def realized_vol(symbol: str = "IWM", lookback: int = 21) -> float | None:
    with engine.connect() as con:
        df = pd.read_sql_query(text("""
            SELECT ts, COALESCE(adj_close, close) AS px
            FROM daily_bars WHERE symbol=:s ORDER BY ts DESC LIMIT 252
        """), con, params={'s': symbol}, parse_dates=['ts'])
    if df.empty: return None
    df = df.sort_values('ts')
    ret = df['px'].pct_change()
    return float(ret.rolling(lookback).std().iloc[-1]) if ret.notna().sum() >= lookback else None

def liquidity_breadth(lookback: int = 21) -> float | None:
    with engine.connect() as con:
        df = pd.read_sql_query(text("SELECT adv_usd_20 FROM universe WHERE included=TRUE"), con)
    if df.empty: return None
    return float(np.nanmedian(df['adv_usd_20']))

def classify_regime(vol: float | None, liq: float | None) -> str:
    if vol is None or liq is None:
        return "unknown"
    # Simple thresholds; can be replaced by quantile-based states
    if vol > 0.03 and liq < 1e6:
        return "stressed"
    if vol < 0.015 and liq > 5e6:
        return "calm"
    return "normal"

def gate_blend_weights(blend: dict[str, float], regime: str) -> dict[str, float]:
    """Adjust model blend weights by regime. Conservative default: more linear bias in stress."""
    if not blend: return blend
    out = blend.copy()
    if regime == "stressed":
        # reduce tree-based complexity, favor ridge
        out = {k: v for k, v in out.items()}
        if 'xgb' in out: out['xgb'] *= 0.6
        if 'rf' in out:  out['rf']  *= 0.6
        if 'ridge' in out: out['ridge'] *= 1.4
    elif regime == "calm":
        # slightly favor xgb in calm
        if 'xgb' in out: out['xgb'] *= 1.2
        if 'rf' in out:  out['rf']  *= 1.0
        if 'ridge' in out: out['ridge'] *= 0.8
    # renormalize
    s = sum(out.values()) or 1.0
    out = {k: v/s for k, v in out.items()}
    return out