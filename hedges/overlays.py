from __future__ import annotations
import pandas as pd, numpy as np
from datetime import date
from sqlalchemy import text
from db import engine, upsert_dataframe, OptionOverlay
from hedges.options import PutHedgeSpec, choose_put_strike

def _latest_book(as_of: date | None = None) -> pd.Series:
    as_of = as_of or pd.Timestamp('today').normalize().date()
    with engine.connect() as con:
        ts = con.execute(text("SELECT MAX(ts) FROM target_positions")).scalar()
        if not ts: return pd.Series(dtype=float)
        df = pd.read_sql_query(text("SELECT symbol, weight FROM target_positions WHERE ts=:t"),
                               con, params={'t': ts})
    return df.set_index('symbol')['weight'] if not df.empty else pd.Series(dtype=float)

def propose_collars(as_of: date | None = None, spec: PutHedgeSpec | None = None) -> int:
    as_of = as_of or pd.Timestamp('today').normalize().date()
    spec = spec or PutHedgeSpec()
    w = _latest_book(as_of)
    if w.empty:
        return 0
    # focus on top weights
    picks = w[w>0].sort_values(ascending=False).head(10)
    rows = []
    # Load prices
    with engine.connect() as con:
        df_px = pd.read_sql_query(text("""
            WITH latest AS (SELECT symbol, MAX(ts) ts FROM daily_bars WHERE symbol IN :syms GROUP BY symbol)
            SELECT b.symbol, COALESCE(b.adj_close, b.close) AS px
            FROM daily_bars b JOIN latest l ON b.symbol=l.symbol AND b.ts=l.ts
        """), con, params={'syms': tuple(picks.index.tolist())})
    px = df_px.set_index('symbol')['px'] if not df_px.empty else pd.Series(dtype=float)
    r_annual = 0.015  # small discount rate
    for s, wt in picks.items():
        S = float(px.get(s, np.nan))
        if not np.isfinite(S) or S <= 0: continue
        Kp = choose_put_strike(S, r_annual, spec)
        # Call strike ~ +10% OTM to partly fund
        Kc = S * 1.10
        # crude premium pct (use IV fallback)
        prem_pct = 0.01 * spec.tenor_days / 30.0  # placeholder; replace with IV-based pricing when available
        rows.append({'as_of': as_of, 'symbol': s, 'strategy': 'collar', 'tenor_days': spec.tenor_days,
                     'put_strike': float(Kp), 'call_strike': float(Kc), 'est_premium_pct': prem_pct,
                     'notes': 'v17 collar proposal'})
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    upsert_dataframe(df, OptionOverlay, ['as_of','symbol','strategy'])
    return int(len(df))
