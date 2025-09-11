from __future__ import annotations
import os
import math
import pandas as pd
from sqlalchemy import create_engine, text
from data.fundamentals import fetch_fundamentals_for_universe

MAP_COLS = {
    "symbol": "symbol",
    "market_cap": "market_cap",
    "shares_outstanding": "shares_outstanding",
    "shares_out": "shares_outstanding"  # fallback if API uses shares_out
}

def main():
    df = fetch_fundamentals_for_universe()
    if df is None or df.empty:
        print("No fundamentals fetched.")
        return 1
    # Normalize columns
    df_norm = {}
    for src, dest in MAP_COLS.items():
        if src in df.columns:
            df_norm[dest] = df[src]
    if "symbol" not in df_norm:
        raise RuntimeError("Fundamentals missing symbol column.")
    out = pd.DataFrame(df_norm)
    out = out.drop_duplicates(subset=["symbol"])
    # Clean
    for col in ["market_cap","shares_outstanding"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x))
    db = os.getenv("DATABASE_URL")
    if db.startswith("postgres://"):
        db = db.replace("postgres://","postgresql+psycopg://",1)
    eng = create_engine(db)
    with eng.begin() as c:
        for row in out.itertuples():
            params = {"symbol": row.symbol,
                      "market_cap": getattr(row,"market_cap", None),
                      "shares_outstanding": getattr(row,"shares_outstanding", None)}
            c.execute(text("""
                UPDATE universe
                   SET market_cap = COALESCE(:market_cap, market_cap),
                       shares_outstanding = COALESCE(:shares_outstanding, shares_outstanding)
                 WHERE symbol = :symbol
            """), params)
    print(f"Upserted fundamentals into universe for {len(out)} symbols.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
