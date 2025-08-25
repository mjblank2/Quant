from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass
class PutHedgeSpec:
    target_delta: float = -0.25   # protective put ~25 delta
    tenor_days: int = 30
    notional_pct: float = 0.20    # hedge ~20% of equity by default
    vol_annual: float = 0.35      # assumed IV if unavailable

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black_scholes_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, -1.0
    d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    put_price = K*math.exp(-r*T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    delta = -_norm_cdf(-d1)
    return put_price, delta

def choose_put_strike(S: float, r_annual: float, spec: PutHedgeSpec) -> float:
    """Crude solve for K such that delta ~ target_delta using BS with assumed IV."""
    T = spec.tenor_days / 252.0
    r = r_annual
    sigma = spec.vol_annual
    # binary search on moneyness
    lo, hi = 0.5*S, S
    for _ in range(30):
        K = 0.5*(lo+hi)
        _, d = black_scholes_put(S, K, T, r, sigma)
        if d < spec.target_delta:  # delta more negative than target => move strike up
            hi = K
        else:
            lo = K
    return 0.5*(lo+hi)

def hedge_budget_notional(equity: float, spec: PutHedgeSpec) -> float:
    return equity * spec.notional_pct
===== END FILE =====