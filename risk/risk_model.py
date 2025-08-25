from __future__ import annotations
import numpy as np, pandas as pd
from sqlalchemy import text
from db import engine

def neutralize_with_sectors(pred: pd.Series, factors: pd.DataFrame | None, sector_dummies: pd.DataFrame | None) -> pd.Series:
    """Cross-sectional neutralization of predictions versus factors (+ optional sector buckets)."""
    X = pd.DataFrame(index=pred.index)
    if factors is not None and not factors.empty:
        X = pd.concat([X, factors.reindex(pred.index)], axis=1)
    if sector_dummies is not None and not sector_dummies.empty:
        sd = sector_dummies.reindex(pred.index).fillna(0.0)
        sd = sd.loc[:, sd.std(axis=0) > 1e-12]  # drop constant columns
        X = pd.concat([X, sd], axis=1)
    if X.empty:
        return pred.fillna(0.0)
    X = X.apply(lambda c: c.fillna(c.median()), axis=0)
    X['__int__'] = 1.0
    y = pred.fillna(0.0).values.reshape(-1,1)
    A = X.values
    AT = A.T
    ATA = AT @ A
    lam = 1e-6 * np.eye(ATA.shape[0])
    beta = np.linalg.pinv(ATA + lam) @ AT @ y
    resid = pred - pd.Series((A @ beta).flatten(), index=pred.index)
    # re-standardize residuals
    r = resid - resid.mean()
    s = resid.std(ddof=0)
    if s > 0:
        r = r / s
    return r

def est_beta_asof(symbols: list[str], as_of, market_symbol: str = "IWM", lookback: int = 63) -> pd.Series:
    """Estimate rolling market beta per symbol using last `lookback` trading days up to `as_of`."""
    if not symbols:
        return pd.Series(dtype=float)
    # Load prices
    sql = """
        SELECT symbol, ts, COALESCE(adj_close, close) AS px
        FROM daily_bars
        WHERE symbol IN :syms OR symbol=:mkt
        ORDER BY symbol, ts
    """
    params = {'syms': tuple(symbols), 'mkt': market_symbol}
    df = pd.read_sql_query(text(sql).bindparams(), engine, params=params, parse_dates=['ts'])
    if df.empty or market_symbol not in df['symbol'].unique():
        return pd.Series(index=symbols, dtype=float)
    # returns
    df['ret'] = df.groupby('symbol')['px'].pct_change()
    as_of = pd.to_datetime(as_of)
    # Filter window
    cutoff = as_of.normalize() - pd.Timedelta(days=lookback*2.5)
    df = df[df['ts'] <= as_of].copy()
    df = df[df['ts'] >= cutoff]
    mkt = df[df['symbol']==market_symbol][['ts','ret']].rename(columns={'ret':'mret'}).dropna()
    out = []
    for s, g in df[df['symbol']!=market_symbol].groupby('symbol'):
        gg = g[['ts','ret']].merge(mkt, on='ts', how='inner').dropna()
        if len(gg) < max(20, lookback//2):
            out.append((s, np.nan)); continue
        cov = gg['ret'].rolling(lookback).cov(gg['mret'])
        var = gg['mret'].rolling(lookback).var()
        beta = (cov/var).iloc[-1] if var.iloc[-1] and np.isfinite(var.iloc[-1]) else np.nan
        out.append((s, float(beta) if beta is not None and np.isfinite(beta) else np.nan))
    return pd.Series(dict(out)).reindex(symbols)

def portfolio_beta(weights: pd.Series, as_of, market_symbol: str = "IWM", lookback: int = 63) -> float:
    betas = est_beta_asof(weights.index.tolist(), as_of, market_symbol, lookback).fillna(0.0)
    return float((weights.reindex(betas.index).fillna(0.0) * betas).sum())
