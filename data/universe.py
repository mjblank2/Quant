from __future__ import annotations
from datetime import datetime
import pandas as pd
from sqlalchemy import text
from db import engine, upsert_dataframe, Universe
from config import MARKET_CAP_MAX, ADV_USD_MIN, ADV_LOOKBACK, APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL

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

def _market_cap_and_adv_yf(symbols: list[str]) -> pd.DataFrame:
    import yfinance as yf
    out = []
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            mc = None
            try:
                mc = getattr(t, 'fast_info', {}).get('market_cap', None)
            except Exception:
                pass
            if mc is None:
                info = t.info or {}
                mc = info.get('marketCap')
            hist = t.history(period='40d', interval='1d', auto_adjust=True)
            if hist is None or hist.empty or len(hist) < 20:
                adv_usd_20 = None
            else:
                dv = (hist['Close'] * hist['Volume']).rolling(20).mean().iloc[-1]
                adv_usd_20 = float(dv) if pd.notnull(dv) else None
            out.append({'symbol': sym, 'market_cap': mc, 'adv_usd_20': adv_usd_20})
        except Exception:
            out.append({'symbol': sym, 'market_cap': None, 'adv_usd_20': None})
    return pd.DataFrame(out)

def rebuild_universe() -> pd.DataFrame:
    base = _list_alpaca_assets()
    if base.empty:
        raise RuntimeError("No assets returned from Alpaca; check credentials/entitlements.")
    enrich = _market_cap_and_adv_yf(base['symbol'].tolist())
    df = base.merge(enrich, on='symbol', how='left')
    df['include_mc'] = df['market_cap'].fillna(0) < MARKET_CAP_MAX
    df['include_adv'] = df['adv_usd_20'].fillna(0) > ADV_USD_MIN
    df['included'] = df['include_mc'] & df['include_adv']
    df['last_updated'] = datetime.utcnow()
    out = df.loc[df['included'], ['symbol','name','exchange','market_cap','adv_usd_20','included','last_updated']].drop_duplicates(subset=['symbol'])
    # Safer refresh: mark existing rows excluded, then upsert new list
    with engine.begin() as con:
        con.execute(text("UPDATE universe SET included = FALSE"))
    upsert_dataframe(out, Universe, ['symbol'])
    return out
