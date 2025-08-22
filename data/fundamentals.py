from __future__ import annotations
from datetime import date
import pandas as pd
from sqlalchemy import text
from db import engine, upsert_dataframe, Fundamentals

def fetch_fundamentals_for_universe(as_of: date | None = None) -> pd.DataFrame:
    import yfinance as yf
    as_of = as_of or pd.Timestamp('today').normalize().date()
    uni = pd.read_sql_query(text("SELECT symbol FROM universe WHERE included = TRUE ORDER BY symbol"), engine)
    symbols = uni['symbol'].tolist()
    rows = []
    for i in range(0, len(symbols), 100):
        subs = symbols[i:i+100]
        tk = yf.Tickers(" ".join(subs))
        for s in subs:
            try:
                t = tk.tickers[s]
                info = t.info or {}
                rows.append({
                    "symbol": s,
                    "as_of": as_of,
                    "pe_ttm": info.get("trailingPE"),
                    "pb": info.get("priceToBook"),
                    "ps_ttm": info.get("priceToSalesTrailing12Months"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "return_on_assets": info.get("returnOnAssets"),
                    "gross_margins": info.get("grossMargins"),
                    "profit_margins": info.get("profitMargins"),
                    "current_ratio": info.get("currentRatio"),
                })
            except Exception:
                rows.append({"symbol": s, "as_of": as_of})
    df = pd.DataFrame(rows)
    upsert_dataframe(df, Fundamentals, ["symbol","as_of"])
    return df

if __name__ == "__main__":
    fetch_fundamentals_for_universe()
