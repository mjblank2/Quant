from __future__ import annotations
from datetime import date, timedelta
from typing import List, Dict, Any
import asyncio
import pandas as pd
from sqlalchemy import text, bindparam
from db import engine, upsert_dataframe, Universe
from config import MARKET_CAP_MAX, ADV_USD_MIN, APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, POLYGON_API_KEY
from utils_http import get_json_async

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
                    if getattr(a, 'class', 'us_equity') in {'us_equity','US_EQUITY'} or getattr(a, 'asset_class', 'us_equity') in {'us_equity','US_EQUITY'}:
                        rows.append({'symbol': a.symbol, 'name': getattr(a, 'name', None), 'exchange': exch})
        except Exception:
            continue
    return pd.DataFrame(rows).drop_duplicates(subset=['symbol'])

async def _poly_ticker_info(symbol: str) -> Dict[str, Any]:
    if not POLYGON_API_KEY:
        return {}
    url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
    return await get_json_async(url, params={"apiKey": POLYGON_API_KEY})

async def _poly_adv(symbol: str, start: date, end: date) -> float | None:
    if not POLYGON_API_KEY:
        return None
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    js = await get_json_async(url, params={"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY})
    try:
        results = js.get("results") or []
        if not results:
            return None
        import numpy as np
        c = pd.Series([r.get("c") for r in results], dtype="float64")
        v = pd.Series([r.get("v") for r in results], dtype="float64")
        dv = (c * v).rolling(20).mean().iloc[-1]
        return float(dv) if pd.notnull(dv) else None
    except Exception:
        return None

async def _enrich_polygon(symbols: List[str]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["symbol","market_cap","adv_usd_20"])
    today = pd.Timestamp("today").normalize().date()
    start = today - timedelta(days=45)
    tasks_info = [_poly_ticker_info(s) for s in symbols]
    tasks_adv  = [_poly_adv(s, start, today) for s in symbols]
    infos = await asyncio.gather(*tasks_info)
    advs  = await asyncio.gather(*tasks_adv)
    rows = []
    for s, info, adv in zip(symbols, infos, advs):
        mc = None
        try:
            res = (info or {}).get("results") or {}
            mc = res.get("market_cap")
        except Exception:
            mc = None
        rows.append({"symbol": s, "market_cap": mc, "adv_usd_20": adv})
    return pd.DataFrame(rows)

def _enrich_universe(symbols: List[str]) -> pd.DataFrame:
    if POLYGON_API_KEY:
        try:
            return asyncio.run(_enrich_polygon(symbols))
        except Exception:
            pass
    # Fallback: set None (you can add yfinance fallback if desired)
    return pd.DataFrame([{"symbol": s, "market_cap": None, "adv_usd_20": None} for s in symbols])

def rebuild_universe() -> pd.DataFrame:
    base = _list_alpaca_assets()
    if base.empty:
        raise RuntimeError("No assets returned from Alpaca; check credentials/entitlements.")
    enrich = _enrich_universe(base['symbol'].tolist())
    df = base.merge(enrich, on='symbol', how='left')
    df['include_mc'] = df['market_cap'].fillna(0) < MARKET_CAP_MAX
    df['include_adv'] = df['adv_usd_20'].fillna(0) > ADV_USD_MIN
    df['included'] = df['include_mc'] & df['include_adv']
    df['last_updated'] = pd.Timestamp.utcnow()
    out = df.loc[df['included'], ['symbol','name','exchange','market_cap','adv_usd_20','included','last_updated']].drop_duplicates(subset=['symbol'])

    if out.empty:
        raise RuntimeError("Universe rebuild produced empty set; keeping prior universe.")

    new_syms = tuple(out['symbol'].unique().tolist())
    with engine.begin() as con:
        upsert_dataframe(out, Universe, ['symbol'], conn=con)
        if new_syms:
            stmt = text("UPDATE universe SET included = FALSE WHERE included = TRUE AND symbol NOT IN :syms").bindparams(bindparam("syms", expanding=True))
            con.execute(stmt, {"syms": new_syms})
    return out

if __name__ == "__main__":
    rebuild_universe()
