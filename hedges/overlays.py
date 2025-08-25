from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from sqlalchemy import text
import math
import pandas as pd
from db import engine

DDL = """
CREATE TABLE IF NOT EXISTS option_overlays (
  ts DATE NOT NULL,
  underlier VARCHAR(16) NOT NULL,
  strategy VARCHAR(32) NOT NULL,  -- protective_put, put_spread, collar
  expiry DATE NOT NULL,
  k1 FLOAT NOT NULL,  -- put strike or lower put for collar/spread
  k2 FLOAT,           -- upper put strike (for spread) or call strike (for collar)
  notional_pct FLOAT NOT NULL,
  est_cost FLOAT,     -- rough premium estimate per unit notional
  PRIMARY KEY(ts, underlier, strategy, expiry, k1, COALESCE(k2, -1))
);
CREATE INDEX IF NOT EXISTS ix_option_overlays_ts ON option_overlays(ts);
"""

def _ensure():
    with engine.begin() as con:
        for stmt in DDL.strip().split(';'):
            s = stmt.strip()
            if s:
                con.execute(text(s))

@dataclass
class OverlaySpec:
    underlier: str = "IWM"
    strategy: str = "protective_put"  # or put_spread, collar
    tenor_days: int = 30
    notional_pct: float = 0.20
    call_otm: float = 0.05
    put_otm: float = 0.10

def _latest_px(symbol: str) -> float | None:
    with engine.connect() as con:
        px = con.execute(text("""
            SELECT COALESCE(adj_close, close) FROM daily_bars
            WHERE symbol=:s ORDER BY ts DESC LIMIT 1
        """), {'s': symbol}).scalar()
    return float(px) if px else None

def _rough_iv(symbol: str = "IWM") -> float:
    # if no IV feed, approximate via realized
    import numpy as np
    df = None
    with engine.connect() as con:
        df = pd.read_sql_query(text("""
            SELECT ts, COALESCE(adj_close, close) AS px
            FROM daily_bars WHERE symbol=:s ORDER BY ts DESC LIMIT 252
        """), con, params={'s': symbol}, parse_dates=['ts'])
    if df is None or df.empty: return 0.25
    df = df.sort_values('ts')
    ret = df['px'].pct_change()
    vol = float(ret.rolling(21).std().iloc[-1]) if ret.notna().sum()>=21 else 0.2
    return max(0.15, min(0.7, vol * (252**0.5) * 0.6))

def _bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    # Approx Black-Scholes put using math.erf
    import math
    def n(x): return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))
    d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    put = K*math.exp(-r*T)*(1-n(d2)) - S*(1-n(d1))
    return max(0.0, put)

def propose_overlays(as_of: date, equity: float, spec: OverlaySpec = OverlaySpec()):
    _ensure()
    S = _latest_px(spec.underlier)
    if not S: return 0
    T = spec.tenor_days/252.0
    r = 0.02
    sigma = _rough_iv(spec.underlier)

    rows = []

    # Protective put
    Kp = S*(1-spec.put_otm)
    cost = _bs_put_price(S, Kp, T, r, sigma)
    rows.append({'ts': as_of, 'underlier': spec.underlier, 'strategy': 'protective_put',
                 'expiry': (as_of + timedelta(days=spec.tenor_days)), 'k1': Kp, 'k2': None,
                 'notional_pct': spec.notional_pct, 'est_cost': cost/spec.put_otm if S>0 else None})

    # Put spread
    Kp_low = S*(1-spec.put_otm)
    Kp_high = S*(1-spec.put_otm/2.0)
    cost_spread = max(0.0, _bs_put_price(S, Kp_low, T, r, sigma) - _bs_put_price(S, Kp_high, T, r, sigma))
    rows.append({'ts': as_of, 'underlier': spec.underlier, 'strategy': 'put_spread',
                 'expiry': (as_of + timedelta(days=spec.tenor_days)), 'k1': Kp_low, 'k2': Kp_high,
                 'notional_pct': spec.notional_pct, 'est_cost': cost_spread/spec.put_otm if S>0 else None})

    # Collar (sell OTM call to fund put)
    Kc = S*(1+spec.call_otm)
    put_cost = _bs_put_price(S, Kp, T, r, sigma)
    # Rough call price parity (very rough; use put as proxy)
    call_price = max(0.0, (put_cost))
    net_cost = max(0.0, put_cost - call_price)
    rows.append({'ts': as_of, 'underlier': spec.underlier, 'strategy': 'collar',
                 'expiry': (as_of + timedelta(days=spec.tenor_days)), 'k1': Kp, 'k2': Kc,
                 'notional_pct': spec.notional_pct, 'est_cost': net_cost/spec.put_otm if S>0 else None})

    df = pd.DataFrame(rows)
    with engine.begin() as con:
        for _, r_ in df.iterrows():
            con.execute(text("""
                INSERT INTO option_overlays(ts, underlier, strategy, expiry, k1, k2, notional_pct, est_cost)
                VALUES (:ts,:u,:st,:exp,:k1,:k2,:np,:c)
                ON CONFLICT(ts, underlier, strategy, expiry, k1, COALESCE(k2,-1)) DO UPDATE
                SET notional_pct=EXCLUDED.notional_pct, est_cost=EXCLUDED.est_cost
            """), {'ts': r_['ts'], 'u': r_['underlier'], 'st': r_['strategy'], 'exp': r_['expiry'],
                    'k1': float(r_['k1']), 'k2': float(r_['k2']) if pd.notna(r_['k2']) else None,
                    'np': float(r_['notional_pct']), 'c': float(r_['est_cost']) if pd.notna(r_['est_cost']) else None})
    return len(rows)