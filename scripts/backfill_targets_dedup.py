import os
import pandas as pd
from sqlalchemy import create_engine, text

H = int(os.getenv("TARGET_HORIZON_DAYS", "5"))

def norm(u): return u.replace("postgres://","postgresql+psycopg://",1) if u.startswith("postgres://") else u

db = os.environ.get("DATABASE_URL")
if not db: raise SystemExit("DATABASE_URL not set")
engine = create_engine(norm(db))

dedup_sql = """
WITH ranked AS (
  SELECT symbol,
         ts::date AS dt,
         close,
         ROW_NUMBER() OVER (PARTITION BY symbol, ts::date ORDER BY ts DESC) rn
  FROM daily_bars
)
SELECT symbol, dt AS ts, close AS px
FROM ranked
WHERE rn=1
ORDER BY symbol, ts;
"""

print("[INFO] Loading deduplicated price series...")
with engine.connect() as c:
    px = pd.read_sql_query(text(dedup_sql), c, parse_dates=["ts"])

if px.empty:
    print("[ERROR] No prices.")
    raise SystemExit(1)

px = px.sort_values(["symbol","ts"]).copy()
px["px_fwd"] = px.groupby("symbol")["px"].shift(-H)
px["fwd_ret"] = (px["px_fwd"]/px["px"]) - 1.0

bench = next((s for s in ("IWM","SPY") if s in px.symbol.unique()), None)
if bench:
    b = px.loc[px.symbol==bench, ["ts","fwd_ret"]].rename(columns={"fwd_ret":"mkt"})
    merged = px.merge(b, on="ts", how="left")
    valid = merged.dropna(subset=["fwd_ret","mkt"])
    if len(valid) >= 100:
        X = valid["mkt"].values
        Y = valid["fwd_ret"].values
        denom = (X*X).sum()
        beta = (X*Y).sum()/denom if denom else 0.0
        alpha = Y.mean() - beta*X.mean()
        merged["fwd_ret_resid"] = merged["fwd_ret"] - (alpha + beta*merged["mkt"])
        print(f"[INFO] Residualization alpha={alpha:.6f} beta={beta:.4f}")
    else:
        print("[WARN] Insufficient overlap for residualization; using raw.")
        merged = merged.assign(fwd_ret_resid=merged["fwd_ret"])
else:
    print("[INFO] No benchmark symbol found; residual = raw.")
    merged = px.assign(fwd_ret_resid=px["fwd_ret"])

targets = merged.loc[~merged["fwd_ret"].isna(), ["symbol","ts","fwd_ret","fwd_ret_resid"]]
print(f"[INFO] Target rows prepared: {len(targets)}")

upd_sql = text("""
UPDATE features
SET fwd_ret = :fwd_ret,
    fwd_ret_resid = :fwd_ret_resid
WHERE symbol=:symbol AND ts=:ts
  AND (fwd_ret IS NULL OR fwd_ret_resid IS NULL)
""")
batch=6000
updated=0
with engine.begin() as c:
    for start in range(0, len(targets), batch):
        chunk = targets.iloc[start:start+batch]
        params=[{
            "symbol": r.symbol,
            "ts": r.ts.date(),
            "fwd_ret": float(r.fwd_ret),
            "fwd_ret_resid": float(r.fwd_ret_resid)
        } for r in chunk.itertuples(index=False)]
        if params:
            c.execute(upd_sql, params)
            updated += len(params)
print(f"[INFO] Attempted updates: {updated}")

with engine.connect() as c:
    counts = c.execute(text("""
      SELECT COUNT(*) AS total,
             COUNT(*) FILTER (WHERE fwd_ret IS NOT NULL) AS fwd_notnull,
             COUNT(*) FILTER (WHERE fwd_ret_resid IS NOT NULL) AS resid_notnull
      FROM features
    """)).first()
print("[INFO] Post-backfill counts:", counts)
