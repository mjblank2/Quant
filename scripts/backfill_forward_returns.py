#!/usr/bin/env python
"""
Backfill forward returns (fwd_ret) and residualized forward returns (fwd_ret_resid)
into the existing `features` table.

Logic:
1. Load all prices from daily_bars.
2. Compute forward return over TARGET_HORIZON_DAYS.
3. Merge into features (symbol, ts).
4. Compute benchmark forward returns (IWM or SPY).
5. Run a single-factor regression fwd_ret ~ fwd_ret_mkt to get alpha/beta and store residual.
6. Batch update features rows where targets are currently NULL.

Safe to re-run; rows already populated are skipped unless --force is passed.

Usage:
    python scripts/backfill_forward_returns.py
    python scripts/backfill_forward_returns.py --force
"""
from __future__ import annotations
import os
import math
import argparse
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DEFAULT_BENCH_CANDIDATES = ["IWM", "SPY"]

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.price_utils import select_price_as

def _norm_url(db: str) -> str:
    if db.startswith("postgres://"):
        return db.replace("postgres://", "postgresql+psycopg://", 1)
    return db

def load_prices(engine) -> pd.DataFrame:
    sql = f"""
        SELECT symbol, ts, {select_price_as('px')}
        FROM daily_bars
        ORDER BY symbol, ts
    """
    return pd.read_sql_query(text(sql), engine, parse_dates=["ts"])

def compute_forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if prices.empty:
        return prices.assign(fwd_ret=np.nan)
    prices = prices.sort_values(["symbol", "ts"]).copy()
    prices["px_fwd"] = prices.groupby("symbol")["px"].shift(-horizon)
    prices["fwd_ret"] = (prices["px_fwd"] / prices["px"]) - 1.0
    return prices.drop(columns=["px_fwd"])

def pick_benchmark(prices: pd.DataFrame, candidates=DEFAULT_BENCH_CANDIDATES) -> str | None:
    syms = set(prices["symbol"].unique())
    for c in candidates:
        if c in syms:
            return c
    return None

def compute_residuals(merged: pd.DataFrame, horizon: int, bench_symbol: str | None) -> pd.DataFrame:
    if bench_symbol is None:
        merged["fwd_ret_resid"] = merged["fwd_ret"]
        return merged
    bench = merged.loc[merged["symbol"] == bench_symbol, ["ts", "fwd_ret"]].rename(columns={"fwd_ret": "fwd_ret_mkt"})
    bench = bench.dropna(subset=["fwd_ret_mkt"])
    merged = merged.merge(bench, on="ts", how="left")
    valid = merged.dropna(subset=["fwd_ret", "fwd_ret_mkt"])
    if len(valid) < 100:
        # Not enough data for stable regression
        merged["fwd_ret_resid"] = merged["fwd_ret"]
        return merged
    X = valid["fwd_ret_mkt"].values
    Y = valid["fwd_ret"].values
    denom = (X * X).sum()
    beta = (X * Y).sum() / denom if denom != 0 else 0.0
    alpha = Y.mean() - beta * X.mean()
    merged["fwd_ret_resid"] = merged["fwd_ret"] - (alpha + beta * merged["fwd_ret_mkt"])
    # If mkt return missing for a row, fallback to raw fwd_ret
    merged.loc[merged["fwd_ret_resid"].isna(), "fwd_ret_resid"] = merged.loc[merged["fwd_ret_resid"].isna(), "fwd_ret"]
    return merged

def load_existing_target_nulls(engine, force: bool) -> pd.DataFrame:
    # Retrieve only rows needing update unless force
    if force:
        sql = "SELECT symbol, ts FROM features"
    else:
        sql = """
            SELECT symbol, ts
            FROM features
            WHERE fwd_ret IS NULL OR fwd_ret_resid IS NULL
        """
    df = pd.read_sql_query(text(sql), engine, parse_dates=["ts"])
    return df.sort_values(["ts", "symbol"])

def batch_update(engine, df: pd.DataFrame, batch_size=5000):
    if df.empty:
        return 0
    updated = 0
    update_sql = text("""
        UPDATE features
        SET fwd_ret = :fwd_ret,
            fwd_ret_resid = :fwd_ret_resid
        WHERE symbol = :symbol AND ts = :ts
    """)
    with engine.begin() as conn:
        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start:start+batch_size]
            params = [
                dict(symbol=r.symbol,
                     ts=r.ts.date(),
                     fwd_ret=None if (isinstance(r.fwd_ret, float) and math.isnan(r.fwd_ret)) else r.fwd_ret,
                     fwd_ret_resid=None if (isinstance(r.fwd_ret_resid, float) and math.isnan(r.fwd_ret_resid)) else r.fwd_ret_resid)
                for r in chunk.itertuples(index=False)
            ]
            conn.execute(update_sql, params)
            updated += len(chunk)
    return updated

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=int(os.getenv("TARGET_HORIZON_DAYS", "5")))
    ap.add_argument("--force", action="store_true", help="Recompute even if targets already populated.")
    args = ap.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL not set", file=sys.stderr)
        return 1
    engine = create_engine(_norm_url(db_url))

    print(f"[INFO] Loading price history...")
    prices = load_prices(engine)
    if prices.empty:
        print("[ERROR] No prices found in daily_bars.")
        return 1

    print(f"[INFO] Computing forward returns (horizon={args.horizon})...")
    prices = compute_forward_returns(prices, args.horizon)

    print("[INFO] Selecting feature rows needing target backfill...")
    target_keys = load_existing_target_nulls(engine, args.force)
    if target_keys.empty:
        print("[INFO] Nothing to update (all targets present). Use --force to recompute.")
        return 0

    print(f"[INFO] Rows needing update: {len(target_keys)}")

    # Merge features keys with price-derived fwd_ret
    merged = target_keys.merge(prices[["symbol", "ts", "fwd_ret"]], on=["symbol", "ts"], how="left")

    bench_symbol = pick_benchmark(prices)
    print(f"[INFO] Benchmark chosen for residualization: {bench_symbol if bench_symbol else 'NONE (fallback to raw)'}")

    merged = compute_residuals(merged, args.horizon, bench_symbol)

    # Update only rows where at least fwd_ret is newly computed
    to_update = merged

    print(f"[INFO] Performing batch updates...")
    updated = batch_update(engine, to_update)
    print(f"[INFO] Updated {updated} feature rows with targets.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
