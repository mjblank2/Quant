import os
import math
import pandas as pd
from sqlalchemy import create_engine, text

H = int(os.getenv("TARGET_HORIZON_DAYS", "5"))

db = os.environ.get("DATABASE_URL")
if not db:
    raise SystemExit("DATABASE_URL not set")
if db.startswith("postgres://"):
    db = db.replace("postgres://", "postgresql+psycopg://", 1)

engine = create_engine(db)

print("[INFO] Loading prices...")
with engine.connect() as c:
    px = pd.read_sql_query(
        text("SELECT symbol, ts, close AS px FROM daily_bars ORDER BY symbol, ts"),
        c,
        parse_dates=["ts"]
    )

if px.empty:
    print("[ERROR] daily_bars empty.")
    raise SystemExit(1)

px = px.sort_values(["symbol", "ts"]).copy()
px["px_fwd"] = px.groupby("symbol")["px"].shift(-H)
px["fwd_ret"] = (px["px_fwd"] / px["px"]) - 1.0

# Pick benchmark (optional residualization)
bench = next((s for s in ("IWM", "SPY") if s in px.symbol.unique()), None)
if bench:
    b = px.loc[px.symbol == bench, ["ts", "fwd_ret"]].rename(columns={"fwd_ret": "mkt"})
    merged = px.merge(b, on="ts", how="left")
    valid = merged.dropna(subset=["fwd_ret", "mkt"])
    if len(valid) >= 100:
        X = valid["mkt"].values
        Y = valid["fwd_ret"].values
        denom = (X * X).sum()
        beta = (X * Y).sum() / denom if denom else 0.0
        alpha = Y.mean() - beta * X.mean()
        merged["fwd_ret_resid"] = merged["fwd_ret"] - (alpha + beta * merged["mkt"])
    else:
        merged = merged.assign(fwd_ret_resid=merged["fwd_ret"])
else:
    merged = px.assign(fwd_ret_resid=px["fwd_ret"])

targets = merged.loc[~merged["fwd_ret"].isna(), ["symbol", "ts", "fwd_ret", "fwd_ret_resid"]]

print("[INFO] Updating features (only NULL targets)...")
upd_sql = text("""
UPDATE features
SET fwd_ret = :fwd_ret,
    fwd_ret_resid = :fwd_ret_resid
WHERE symbol = :symbol
  AND ts = :ts
  AND (fwd_ret IS NULL OR fwd_ret_resid IS NULL)
""")

batch = 6000
updated = 0
with engine.begin() as c:
    for start in range(0, len(targets), batch):
        chunk = targets.iloc[start:start+batch]
        params = [{
            "symbol": r.symbol,
            "ts": r.ts.date(),
            "fwd_ret": float(r.fwd_ret),
            "fwd_ret_resid": float(r.fwd_ret_resid)
        } for r in chunk.itertuples(index=False)]
        if params:
            c.execute(upd_sql, params)
            updated += len(params)

print(f"[INFO] Attempted row updates: {updated}")

with engine.connect() as c:
    counts = c.execute(text("""
      SELECT
        COUNT(*) AS total,
        COUNT(*) FILTER (WHERE fwd_ret IS NOT NULL) AS fwd_notnull,
        COUNT(*) FILTER (WHERE fwd_ret_resid IS NOT NULL) AS resid_notnull
      FROM features
    """)).first()
    print("[INFO] Post-backfill counts:", counts)
