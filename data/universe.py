from __future__ import annotations
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd
from sqlalchemy import text, bindparam
from db import get_engine, upsert_dataframe, Universe
from config import MARKET_CAP_MAX, ADV_USD_MIN, ADV_LOOKBACK, APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, POLYGON_API_KEY, UNIVERSE_CONCURRENCY
from utils_http import get_json

POLY_BASE = "https://api.polygon.io"

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

def _poly_ticker_details(symbol: str) -> dict:
    url = f"{POLY_BASE}/v3/reference/tickers/{symbol}"
    return get_json(url, {"apiKey": POLYGON_API_KEY}) or {}

def _poly_adv_usd(symbol: str, asof: date, lookback: int = 20) -> float | None:
    start = (pd.Timestamp(asof) - pd.Timedelta(days=int(lookback * 2))).strftime("%Y-%m-%d")
    end = pd.Timestamp(asof).strftime("%Y-%m-%d")
    url = f"{POLY_BASE}/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    js = get_json(url, {"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY, "limit": 50000})
    rows = (js or {}).get("results") or []
    if not rows:
        return None
    df = pd.DataFrame(rows)
    try:
        dv = (df["c"] * df["v"]).rolling(lookback).mean()
        val = float(dv.iloc[-1]) if len(dv) >= lookback and pd.notnull(dv.iloc[-1]) else None
    except Exception:
        val = None
    return val

def _enrich_one(symbol: str, asof: date) -> dict:
    det = _poly_ticker_details(symbol).get("results") or {}
    mc = det.get("market_cap")
    adv = _poly_adv_usd(symbol, asof, ADV_LOOKBACK)
    return {"symbol": symbol, "market_cap": mc, "adv_usd_20": adv}

def rebuild_universe() -> pd.DataFrame:
    base = _list_alpaca_assets()
    if base.empty:
        raise RuntimeError("No assets returned from Alpaca; check credentials/entitlements.")
    asof = pd.Timestamp('today').normalize().date()

    rows = []
    with ThreadPoolExecutor(max_workers=max(2, UNIVERSE_CONCURRENCY)) as ex:
        futs = {ex.submit(_enrich_one, s, asof): s for s in base["symbol"].tolist()}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception:
                rows.append({"symbol": futs[fut], "market_cap": None, "adv_usd_20": None})

    enrich = pd.DataFrame(rows)
    df = base.merge(enrich, on='symbol', how='left')
    df['include_mc'] = df['market_cap'].fillna(0) < MARKET_CAP_MAX
    df['include_adv'] = df['adv_usd_20'].fillna(0) > ADV_USD_MIN
    df['included'] = df['include_mc'] & df['include_adv']
    df['last_updated'] = datetime.utcnow()
    out = df.loc[df['included'], ['symbol','name','exchange','market_cap','adv_usd_20','included','last_updated']].drop_duplicates(subset=['symbol'])

    if out.empty:
        raise RuntimeError("Universe rebuild produced empty set; keeping prior universe unchanged.")

    eng = get_engine()
    # Transaction: upsert new set, then flip those not present
    new_syms = out["symbol"].unique().tolist()
    from sqlalchemy import bindparam
    with eng.begin() as con:
        upsert_dataframe(out, Universe, ['symbol'])
        if new_syms:
            stmt = text("UPDATE universe SET included = FALSE WHERE included = TRUE AND symbol NOT IN :syms").bindparams(bindparam("syms", expanding=True))
            con.execute(stmt, {"syms": tuple(new_syms)})
    return out

if __name__ == "__main__":
    rebuild_universe()

