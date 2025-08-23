from __future__ import annotations
import asyncio
import pandas as pd
from sqlalchemy import text
import aiohttp
from db import engine, upsert_dataframe, Fundamentals
from config import POLYGON_API_KEY, HTTP_TIMEOUT, HTTP_CONCURRENCY
from utils_http_async import get_json_async

def _as_date(x):
    try:
        return pd.to_datetime(x, utc=True).date()
    except Exception:
        return None

async def _poly_financials_async(symbols: list[str], per_symbol_max: int = 12) -> pd.DataFrame:
    if not POLYGON_API_KEY or not symbols:
        return pd.DataFrame(columns=['symbol','as_of','available_at'])

    sem = asyncio.Semaphore(HTTP_CONCURRENCY)

    async def fetch_one(session: aiohttp.ClientSession, s: str):
        async with sem:
            url = "https://api.polygon.io/vX/reference/financials"
            params = {"ticker": s, "limit": min(per_symbol_max, 50), "order": "desc", "apiKey": POLYGON_API_KEY}
            rows = []
            fetched = 0
            next_url = None
            while True:
                js = await get_json_async(session, next_url or url, params=None if next_url else params, timeout=HTTP_TIMEOUT)
                if not js or not js.get("results"):
                    break
                for res in js["results"]:
                    filing_dt = res.get("acceptance_datetime") or res.get("filing_date")
                    try:
                        fdt = pd.to_datetime(filing_dt, utc=True)
                        available_at = (fdt + pd.Timedelta(days=1)).normalize().date()
                    except Exception:
                        available_at = _as_date(res.get("filing_date"))
                    as_of = _as_date(res.get("period_of_report_date")) or _as_date(res.get("fiscal_period_end")) or _as_date(res.get("filing_date"))
                    fin = res.get("financials") or {}
                    inc = fin.get("income_statement") or {}
                    bal = fin.get("balance_sheet") or {}

                    revenue = inc.get("revenues") or inc.get("total_revenue")
                    gross_profit = inc.get("gross_profit")
                    net_income = inc.get("net_income_loss")
                    total_assets = bal.get("assets") or bal.get("total_assets")
                    equity = bal.get("shareholders_equity") or bal.get("total_shareholders_equity")
                    current_assets = bal.get("current_assets")
                    current_liab = bal.get("current_liabilities")
                    debt = bal.get("long_term_debt") or bal.get("debt") or bal.get("total_debt")

                    def _safe_div(a, b):
                        try:
                            a = float(a); b = float(b)
                            if b == 0: return None
                            return a / b
                        except Exception:
                            return None

                    gm = _safe_div(gross_profit, revenue)
                    pm = _safe_div(net_income, revenue)
                    roa = _safe_div(net_income, total_assets)
                    cr  = _safe_div(current_assets, current_liab)
                    de  = _safe_div(debt, equity)

                    rows.append({
                        "symbol": s,
                        "as_of": as_of,
                        "available_at": available_at,
                        "pe_ttm": None,
                        "pb": None,
                        "ps_ttm": None,
                        "debt_to_equity": de,
                        "return_on_assets": roa,
                        "gross_margins": gm,
                        "profit_margins": pm,
                        "current_ratio": cr,
                    })
                    fetched += 1
                    if fetched >= per_symbol_max:
                        break
                next_url = js.get("next_url")
                if next_url and "apiKey" not in next_url:
                    next_url = f"{next_url}&apiKey={POLYGON_API_KEY}"
                if fetched >= per_symbol_max or not next_url:
                    break
            if not rows:
                rows = [{"symbol": s, "as_of": None, "available_at": None}]
            return rows

    async with aiohttp.ClientSession() as session:
        batches = await asyncio.gather(*(fetch_one(session, s) for s in symbols))
    flat = [r for sub in batches for r in (sub if isinstance(sub, list) else [sub])]
    return pd.DataFrame(flat)

def fetch_fundamentals_for_universe() -> pd.DataFrame:
    uni = pd.read_sql_query(text("SELECT symbol FROM universe WHERE included = TRUE ORDER BY symbol"), engine)
    symbols = uni['symbol'].tolist()
    if not symbols:
        return pd.DataFrame(columns=['symbol','as_of','available_at'])
    df = asyncio.run(_poly_financials_async(symbols))
    df = df[~df["available_at"].isna()].copy() if not df.empty else df
    upsert_dataframe(df, Fundamentals, ["symbol","as_of"])
    return df

if __name__ == "__main__":
    fetch_fundamentals_for_universe()
