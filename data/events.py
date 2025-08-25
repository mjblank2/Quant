from __future__ import annotations
import pandas as pd
from sqlalchemy import text
from datetime import timedelta
from db import engine, upsert_dataframe, AltSignal, RussellMembership

# === Earnings (PEAD) loader ===
def load_earnings_csv(path: str) -> int:
    df = pd.read_csv(path)
    # Expected columns: symbol, earnings_date, eps_actual, eps_est, rev_actual, rev_est
    # Normalize types
    df['symbol'] = df['symbol'].str.upper().str.strip()
    df['earnings_date'] = pd.to_datetime(df['earnings_date']).dt.date
    # Surprise magnitudes
    df['pead_surprise_eps'] = (df['eps_actual'] - df['eps_est']) / df['eps_est'].replace(0, pd.NA)
    df['pead_surprise_rev'] = (df['rev_actual'] - df['rev_est']) / df['rev_est'].replace(0, pd.NA)
    # Tradeable at T+1
    df['ts'] = pd.to_datetime(df['earnings_date']).dt.date + timedelta(days=1)
    # Sparse event flag
    pead_ev = df[['symbol','ts']].copy(); pead_ev['name']='pead_event'; pead_ev['value']=1.0
    eps = df[['symbol','ts','pead_surprise_eps']].rename(columns={'pead_surprise_eps':'value'}); eps['name']='pead_surprise_eps'
    rev = df[['symbol','ts','pead_surprise_rev']].rename(columns={'pead_surprise_rev':'value'}); rev['name']='pead_surprise_rev'
    out = pd.concat([pead_ev, eps, rev], ignore_index=True)
    upsert_dataframe(out[['symbol','ts','name','value']], AltSignal, ['symbol','ts','name'])
    return int(len(out))

# === Russell Reconstitution loader ===
def load_russell_membership_csv(path: str) -> int:
    df = pd.read_csv(path)
    # Expected columns: ts,symbol,action (add|drop|keep)
    df['ts'] = pd.to_datetime(df['ts']).dt.date
    df['symbol'] = df['symbol'].str.upper().str.strip()
    df['action'] = df['action'].str.lower().str.strip()
    # Store membership rows
    upsert_dataframe(df[['symbol','ts','action']], RussellMembership, ['symbol','ts'])
    # Also emit sparse signal +1 add, -1 drop, 0 keep
    sig = df.copy()
    mapv = {'add':1.0, 'drop':-1.0, 'keep':0.0}
    sig['value'] = sig['action'].map(mapv).fillna(0.0)
    sig['name'] = 'russell_inout'
    upsert_dataframe(sig[['symbol','ts','name','value']], AltSignal, ['symbol','ts','name'])
    return int(len(df))

if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--earnings_csv", type=str, default=None)
    p.add_argument("--russell_csv", type=str, default=None)
    args = p.parse_args()
    if args.earnings_csv and os.path.exists(args.earnings_csv):
        print("Earnings rows:", load_earnings_csv(args.earnings_csv))
    if args.russell_csv and os.path.exists(args.russell_csv):
        print("Russell rows:", load_russell_membership_csv(args.russell_csv))
