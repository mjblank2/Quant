from __future__ import annotations
import asyncio
import pandas as pd
from sqlalchemy import text, bindparam
from db import engine, upsert_dataframe, Universe
from config import MARKET_CAP_MAX, ADV_USD_MIN, APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, POLYGON_API_KEY, HTTP_TIMEOUT, HTTP_CONCURRENCY
from utils_http_async import get_json_async
import aiohttp

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
                    cls = getattr(a, 'class', getattr(a, 'asset_class', ''))
                    if cls in {'us_equity','US_EQUITY',''}:
                        rows.append({'symbol': a.symbol, 'name': getattr(a, 'name', None), 'exchange': exch})
        except Exception:
            continue
    return pd.DataFrame(rows).drop_duplicates(subset=['symbol'])

async def _poly_adv_mcap_async(symbols: list[str]) -> pd.DataFrame:
    if not POLYGON_API_KEY or not symbols:
        return pd.DataFrame(columns=['symbol','market_cap','adv_usd_20'])
    end = pd.Timestamp.utcnow().normalize().date()
    start = (pd.Timestamp(end) - pd.Timedelta(days=60)).date()
    start_s = pd.Timestamp(start).strftime('%Y-%m-%d')
    end_s = pd.Timestamp(end).strftime('%Y-%m-%d')

    sem = asyncio.Semaphore(HTTP_CONCURRENCY)

    async def fetch_one(session: aiohttp.ClientSession, s: str):
        async with sem:
            aggs_url = f"https://api.polygon.io/v2/aggs/ticker/{s}/range/1/day/{start_s}/{end_s}"
            aggs = await get_json_async(session, aggs_url, params={"adjusted":"true","sort":"desc","limit":120,"apiKey":POLYGON_API_KEY}, timeout=HTTP_TIMEOUT)
            dv20 = None
            last_close = None
            if aggs and aggs.get("results"):
                res = aggs["results"]
                import pandas as _pd
                df = _pd.DataFrame([{"c": r.get("c"), "v": r.get("v"), "t": r.get("t")} for r in res])
                if not df.empty:
                    df["dv"] = df["c"] * df["v"]
                    df = df.sort_values("t")
                    dv20 = float(df["dv"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else None
                    last_close = float(df["c"].iloc[-1])
            ref_url = f"https://api.polygon.io/v3/reference/tickers/{s}"
            ref = await get_json_async(session, ref_url, params={"date": end_s, "apiKey": POLYGON_API_KEY}, timeout=HTTP_TIMEOUT)
            mcap = None
            if ref and ref.get("results"):
                rr = ref["results"]
                mcap = rr.get("market_cap")
                if not mcap and last_close is not None:
                    shares = rr.get("share_class_shares_outstanding") or rr.get("weighted_shares_outstanding")
                    if shares:
                        try:
                            mcap = float(last_close) * float(shares)
                        except Exception:
                            pass
            return {"symbol": s, "market_cap": mcap, "adv_usd_20": dv20}

    async with aiohttp.ClientSession() as session:
        rows = await asyncio.gather(*(fetch_one(session, s) for s in symbols))
    return pd.DataFrame(rows)

def rebuild_universe() -> pd.DataFrame:
    base = _list_alpaca_assets()
    if base.empty:
        raise RuntimeError("No assets from Alpaca; check credentials/entitlements.")
    enrich = asyncio.run(_poly_adv_mcap_async(base['symbol'].tolist()))
    df = base.merge(enrich, on='symbol', how='left')
    df['include_mc'] = df['market_cap'].fillna(0) < MARKET_CAP_MAX
    df['include_adv'] = df['adv_usd_20'].fillna(0) > ADV_USD_MIN
    df['included'] = df['include_mc'] & df['include_adv']
    df['last_updated'] = pd.Timestamp.utcnow()
    out = df.loc[df['included'], ['symbol','name','exchange','market_cap','adv_usd_20','included','last_updated']].drop_duplicates(subset=['symbol'])

    if out.empty:
        raise RuntimeError("Universe rebuild produced empty set; refusing to overwrite existing universe.")

    new_syms = tuple(out["symbol"].unique().tolist())

    with engine.begin() as con:
        upsert_dataframe(out, Universe, ['symbol'], conn=con)
        if new_syms:
            stmt = text("UPDATE universe SET included = FALSE WHERE included = TRUE AND symbol NOT IN :syms").bindparams(bindparam("syms", expanding=True))
            con.execute(stmt, {"syms": new_syms})
    return out

if __name__ == "__main__":
    rebuild_universe()
