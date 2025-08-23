from __future__ import annotations
from datetime import date, timedelta, datetime
from typing import List
import pandas as pd
from sqlalchemy import text, bindparam
from db import engine, upsert_dataframe, Universe
from config import MARKET_CAP_MAX, ADV_USD_MIN, APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, POLYGON_API_KEY
from utils_http import get_json

def _list_alpaca_assets() -> pd.DataFrame:
    import alpaca_trade_api as tradeapi
    api = tradeapi.REST(key_id=APCA_API_KEY_ID, secret_key=APCA_API_SECRET_KEY, base_url=APCA_API_BASE_URL)
    assets = api.list_assets(status='active')
    rows = []
    for a in assets:
        try:
            if getattr(a, 'tradable', False) and getattr(a, 'status', 'active') == 'active':
                exch = getattr(a, 'exchange', '') or getattr(a, 'primary_exchange', '')
                if exch in {'NYSE','NASDAQ','ARCA','BATS','AMEX'}:
                    cls = getattr(a, 'asset_class', '') or getattr(a, 'class', '')
                    if cls.lower() in {'us_equity', 'us_equities', 'us_equity'} or cls.upper() == 'US_EQUITY':
                        rows.append({'symbol': a.symbol, 'name': getattr(a, 'name', None), 'exchange': exch})
        except Exception:
            continue
    return pd.DataFrame(rows).drop_duplicates(subset=['symbol'])

def _polygon_adv_and_shares(symbols: List[str]) -> pd.DataFrame:
    # Compute ADV (USD) from last ~40 trading days of aggregates, and get shares outstanding via reference/financials (latest filing PIT)
    if not POLYGON_API_KEY:
        return pd.DataFrame(columns=['symbol','adv_usd_20','market_cap'])
    rows = []
    end = pd.Timestamp('today').normalize().date()
    start = (pd.Timestamp(end) - pd.Timedelta(days=70)).date()
    for s in symbols:
        try:
            # Aggregates for ADV
            url = f"https://api.polygon.io/v2/aggs/ticker/{s}/range/1/day/{start}/{end}"
            js = get_json(url, params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}, max_tries=6)
            adv = None
            last_close = None
            if js and js.get("results"):
                df = pd.DataFrame(js["results"])
                # c=close, v=volume
                df["dv"] = df["c"] * df["v"]
                adv = float(df["dv"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else None
                last_close = float(df["c"].iloc[-1])
            # Shares outstanding from financials (latest PIT)
            fin_url = "https://api.polygon.io/vX/reference/financials"
            fjs = get_json(fin_url, params={"tickers": s, "limit": 1, "sort": "-filing_date", "apiKey": POLYGON_API_KEY}, max_tries=6)
            shares = None
            if fjs and fjs.get("results"):
                fin = fjs["results"][0]
                shares = fin.get("weighted_average_shares_outstanding_basic") or fin.get("weighted_average_shares_outstanding_diluted") or fin.get("shares_outstanding")
            mc = float(last_close * shares) if (last_close is not None and shares) else None
            rows.append({"symbol": s, "adv_usd_20": adv, "market_cap": mc})
        except Exception:
            rows.append({"symbol": s, "adv_usd_20": None, "market_cap": None})
    return pd.DataFrame(rows)

def rebuild_universe() -> pd.DataFrame:
    base = _list_alpaca_assets()
    if base.empty:
        raise RuntimeError("No assets returned from Alpaca; check credentials/entitlements.")
    enrich = _polygon_adv_and_shares(base['symbol'].tolist())
    df = base.merge(enrich, on='symbol', how='left')
    df['include_mc'] = df['market_cap'].fillna(0) < MARKET_CAP_MAX
    df['include_adv'] = df['adv_usd_20'].fillna(0) > ADV_USD_MIN
    df['included'] = df['include_mc'] & df['include_adv']
    df['last_updated'] = datetime.utcnow()

    out = df.loc[df['included'], ['symbol','name','exchange','market_cap','adv_usd_20','included','last_updated']].drop_duplicates(subset=['symbol'])
    # Safety: do not flip everything to FALSE if out is empty
    if out.empty:
        raise RuntimeError("Universe rebuild produced empty set; keeping prior universe unchanged.")
    new_syms = tuple(out['symbol'].unique().tolist())

    # Atomic upsert + flip
    from sqlalchemy import text, bindparam
    with engine.begin() as con:
        upsert_dataframe(out, Universe, ['symbol'], conn=con)
        if new_syms:
            stmt = text("UPDATE universe SET included = FALSE WHERE included = TRUE AND symbol NOT IN :syms").bindparams(bindparam("syms", expanding=True))
            con.execute(stmt, {"syms": new_syms})

    return out

if __name__ == "__main__":
    rebuild_universe()
