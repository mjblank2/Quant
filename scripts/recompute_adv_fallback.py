from __future__ import annotations
import os, pandas as pd
from sqlalchemy import create_engine, text

def main():
    db = os.getenv("DATABASE_URL")
    if not db:
        raise SystemExit("DATABASE_URL not set")
    if db.startswith("postgres://"):
        db = db.replace("postgres://","postgresql+psycopg://",1)
    eng = create_engine(db)
    with eng.connect() as c:
        bars = pd.read_sql_query(text("""
            SELECT symbol, ts, close, volume
            FROM daily_bars
            WHERE ts >= (SELECT MAX(ts) - INTERVAL '90 days' FROM daily_bars)
        """), c, parse_dates=['ts'])
    if bars.empty:
        print("No bars found.")
        return 0
    bars = bars.dropna(subset=["close","volume"])
    bars["dollar_volume"] = bars["close"] * bars["volume"]
    bars = bars.sort_values(["symbol","ts"])
    # Rolling 20, but allow fallback if <20 bars: use mean of all available
    def adv20(group):
        dv = group.set_index("ts")["dollar_volume"]
        r = dv.rolling(20, min_periods=1).mean()
        return r

    adv = bars.groupby("symbol", group_keys=False).apply(adv20).reset_index()
    adv = adv.rename(columns={"dollar_volume":"adv_usd_20"})
    latest_adv = adv.groupby("symbol").tail(1)
    print("Computed ADV rows:", len(latest_adv))
    with eng.begin() as c:
        for row in latest_adv.itertuples():
            c.execute(text("""
               UPDATE universe
                  SET adv_usd_20 = :adv
                WHERE symbol = :sym
            """), {"adv": float(row.adv_usd_20), "sym": row.symbol})
    print("Updated universe.adv_usd_20 (fallback version).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
