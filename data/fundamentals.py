from __future__ import annotations
from datetime import date
import pandas as pd
from sqlalchemy import text
from db import get_engine, upsert_dataframe, Fundamentals
from config import POLYGON_API_KEY
from utils_http import get_json

POLY_BASE = "https://api.polygon.io"

def _safe_div(num, den):
    try:
        if num is None or den in (None, 0):
            return None
        return float(num) / float(den) if den else None
    except Exception:
        return None

def _poly_financials_latest(symbol: str) -> tuple[dict, date | None]:
    url = f"{POLY_BASE}/vX/reference/financials"
    params = {"ticker": symbol, "limit": 1, "sort": "period_of_report_date", "order": "desc", "apiKey": POLYGON_API_KEY}
    js = get_json(url, params, timeout=25, max_tries=5) or {}
    res = (js.get("results") or [])
    if not res:
        return {}, None
    r = res[0]
    # Determine the as-of date from multiple possible fields
    period = r.get("period_of_report_date") or r.get("fiscal_period_end") or r.get("filing_date") or r.get("start_date") or r.get("end_date")
    as_of = pd.to_datetime(period).date() if period else None
    return r, as_of

def fetch_fundamentals_for_universe(as_of: date | None = None) -> pd.DataFrame:
    eng = get_engine()
    uni = pd.read_sql_query(text("SELECT symbol FROM universe WHERE included = TRUE ORDER BY symbol"), eng)
    symbols = uni['symbol'].tolist()
    rows = []
    for s in symbols:
        r, aof = _poly_financials_latest(s)
        fs = (r.get("financials") or {})
        isec = fs.get("income_statement") or {}
        bsec = fs.get("balance_sheet") or {}
        revenue = (isec.get("revenues") or {}).get("value")
        net_income = (isec.get("net_income_loss") or {}).get("value") or (isec.get("net_income") or {}).get("value")
        gross_profit = (isec.get("gross_profit") or {}).get("value")
        total_assets = (bsec.get("assets") or {}).get("value")
        equity = (bsec.get("equity_attributable_to_parent") or {}).get("value") or (bsec.get("shareholders_equity") or {}).get("value")
        total_debt = ((bsec.get("long_term_debt") or {}).get("value") or 0) + ((bsec.get("short_term_debt") or {}).get("value") or 0)
        current_assets = (bsec.get("current_assets") or {}).get("value")
        current_liabilities = (bsec.get("current_liabilities") or {}).get("value")
        gm = _safe_div(gross_profit, revenue)
        pm = _safe_div(net_income, revenue)
        roa = _safe_div(net_income, total_assets)
        de = _safe_div(total_debt, equity)
        cr = _safe_div(current_assets, current_liabilities)
        rows.append({
            "symbol": s, "as_of": aof or (as_of or pd.Timestamp('today').normalize().date()),
            "pe_ttm": None, "pb": None, "ps_ttm": None,
            "debt_to_equity": de, "return_on_assets": roa, "gross_margins": gm,
            "profit_margins": pm, "current_ratio": cr
        })
    df = pd.DataFrame(rows)
    upsert_dataframe(df, Fundamentals, ["symbol","as_of"])
    return df

if __name__ == "__main__":
    fetch_fundamentals_for_universe()

