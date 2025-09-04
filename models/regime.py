from __future__ import annotations
import numpy as np
import pandas as pd
from sqlalchemy import text
from db import engine

def _load_series(symbol: str, end_ts: pd.Timestamp, lookback: int = 252) -> pd.Series:
    with engine.connect() as con:
        df = pd.read_sql_query(text(\"\"
            SELECT ts, COALESCE(adj_close, close) AS px FROM daily_bars
            WHERE symbol=:s AND ts<=:e ORDER BY ts DESC LIMIT :lim
        \"\"), con, params={'s': symbol, 'e': end_ts.date(), 'lim': lookback+5}, parse_dates=['ts']).sort_values('ts')
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index('ts')['px'].pct_change().dropna()

def classify_regime(as_of, market_symbol: str = "IWM") -> str:
    as_of = pd.to_datetime(as_of)
    r = _load_series(market_symbol, as_of, 252)
    if r.empty:
        return "normal"
    vol21 = float(r.rolling(21).std().iloc[-1] or 0.0)
    with engine.connect() as con:
        df = pd.read_sql_query(text(\"SELECT vol_21 FROM features WHERE ts=(SELECT MAX(ts) FROM features WHERE ts<=:asof)\"),
                               con, params={'asof': as_of.date()})
    breadth = float(df['vol_21'].median()) if not df.empty else 0.02
    if vol21 > 0.03 or breadth > 0.03:
        return "stressed"
    if vol21 < 0.015 and breadth < 0.02:
        return "calm"
    return "normal"

def gate_blend_weights(blend: dict[str,float], regime: str) -> dict[str,float]:
    w = dict(blend)
    if not w:
        return w
    if regime == "stressed":
        if 'rf' in w: w['rf'] *= 0.5
        if 'xgb' in w: w['xgb'] *= 0.7
        if 'ridge' in w: w['ridge'] *= 1.3
    elif regime == "calm":
        if 'xgb' in w: w['xgb'] *= 1.15
    s = sum(max(0.0, v) for v in w.values()) or 1.0
    return {k: max(0.0, v)/s for k, v in w.items()}
