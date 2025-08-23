from __future__ import annotations
import asyncio
from datetime import date
import numpy as np
import pandas as pd
from sqlalchemy import text
from db import engine, upsert_dataframe, Fundamentals
from config import POLYGON_API_KEY
from utils_http_async import get_json_async

POLY_FIN_URL = "https://api.polygon.io/vX/reference/financials"

def _val(d: dict, *keys):
    for k in keys:
        if d is None:
            return None
        v = d.get(k)
        if v is not None:
            return v
    return None

async def _fetch_symbol_fin(client, symbol: str) -> dict:
    params = {"tickers": symbol, "limit": 1, "sort": "-filing_date", "apiKey": POLYGON_API_KEY}
    js = await get_json_async(client, POLY_FIN_URL, params=params, max_tries=6, backoff_base=0.6, timeout=30.0)
    out = {"symbol": symbol}
    out["_denom_net_income"] = None
    out["_denom_equity"] = None
    out["_denom_revenue"] = None
    try:
        if not js or not js.get("results"):
            return out
        r = js["results"][0]
        fin = r.get("financials", {}) or {}
        inc = fin.get("income_statement", {}) or {}
        bal = fin.get("balance_sheet", {}) or {}
        cfs = fin.get("cash_flow_statement", {}) or {}
        as_of_src = r.get("period_of_report_date") or r.get("fiscal_period_end") or r.get("filing_date") or r.get("end_date")
        as_of = pd.to_datetime(as_of_src).date() if as_of_src else pd.Timestamp('today').normalize().date()

        # Extract fields (robust to missing)
        shares_equity = _val(bal, "stockholders_equity")
        revenue = _val(inc, "revenues") or _val(inc, "revenue")
        gross_profit = _val(inc, "gross_profit")
        net_income = _val(inc, "net_income_loss") or _val(inc, "net_income")
        current_assets = _val(bal, "current_assets")
        current_liabilities = _val(bal, "current_liabilities")
        total_assets = _val(bal, "assets")
        total_liabilities = _val(bal, "liabilities")

        out.update({
            "as_of": as_of,
            "pe_ttm": None,  # computed later with market cap denominator if possible
            "pb": None,
            "ps_ttm": None,
            "debt_to_equity": (float(total_liabilities)/float(shares_equity)) if (total_liabilities and shares_equity and float(shares_equity) != 0) else None,
            "return_on_assets": (float(net_income)/float(total_assets)) if (net_income and total_assets and float(total_assets) != 0) else None,
            "gross_margins": (float(gross_profit)/float(revenue)) if (gross_profit and revenue and float(revenue) != 0) else None,
            "profit_margins": (float(net_income)/float(revenue)) if (net_income and revenue and float(revenue) != 0) else None,
            "current_ratio": (float(current_assets)/float(current_liabilities)) if (current_assets and current_liabilities and float(current_liabilities) != 0) else None,
        })

        out["_denom_revenue"] = revenue
        out["_denom_net_income"] = net_income
        out["_denom_equity"] = shares_equity
        return out
    except Exception:
        return out

async def _fetch_all(symbols: list[str], concurrency: int = 12) -> list[dict]:
    import httpx
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        async def _task(sym: str):
            async with sem:
                return await _fetch_symbol_fin(client, sym)
        return await asyncio.gather(*[_task(s) for s in symbols])

def fetch_fundamentals_for_universe(as_of: date | None = None) -> pd.DataFrame:
    if not POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY is required for fundamentals.")
    uni = pd.read_sql_query(text("SELECT symbol, market_cap FROM universe WHERE included = TRUE ORDER BY symbol"), engine)
    symbols = uni['symbol'].tolist()
    if not symbols:
        return pd.DataFrame(columns=["symbol","as_of"])

    rows = asyncio.run(_fetch_all(symbols))
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Compute PE/PB/PS via universe market cap, if denominators present
    mc_map = uni.set_index("symbol")["market_cap"].to_dict()

    def safe_float(x):
        try:
            return float(x) if x is not None and np.isfinite(float(x)) else None
        except Exception:
            return None
    def ratio(mc, denom):
        try:
            mc = safe_float(mc); denom = safe_float(denom)
            if mc is None or denom is None or denom == 0:
                return None
            return mc / denom
        except Exception:
            return None

    df["pe_ttm"] = [ratio(mc_map.get(s), r) for s, r in zip(df["symbol"], df.get("_denom_net_income"))]
    df["ps_ttm"] = [ratio(mc_map.get(s), r) for s, r in zip(df["symbol"], df.get("_denom_revenue"))]
    df["pb"] = [ratio(mc_map.get(s), r) for s, r in zip(df["symbol"], df.get("_denom_equity"))]

    # Drop temp denom cols
    for c in ["_denom_net_income","_denom_equity","_denom_revenue"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    upsert_dataframe(df[["symbol","as_of","pe_ttm","pb","ps_ttm","debt_to_equity","return_on_assets","gross_margins","profit_margins","current_ratio"]], Fundamentals, ["symbol","as_of"])
    return df

if __name__ == "__main__":
    fetch_fundamentals_for_universe()
