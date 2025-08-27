# data/universe.py
from __future__ import annotations

from datetime import date, timedelta
from typing import List, Dict, Any
import os
import asyncio
import pandas as pd
from sqlalchemy import text, bindparam

from db import engine, upsert_dataframe, Universe

# ---------------------------------------------------------------------
# Robust config import with sane fallbacks so cron never dies at import
# ---------------------------------------------------------------------
try:
    from config import (
        MARKET_CAP_MAX, ADV_USD_MIN,
        APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL,
        POLYGON_API_KEY,
    )
except Exception:
    MARKET_CAP_MAX       = float(os.getenv("MARKET_CAP_MAX", "3000000000"))  # $3B default
    ADV_USD_MIN          = float(os.getenv("ADV_USD_MIN", "25000"))
    APCA_API_KEY_ID      = os.getenv("APCA_API_KEY_ID")
    APCA_API_SECRET_KEY  = os.getenv("APCA_API_SECRET_KEY")
    APCA_API_BASE_URL    = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    POLYGON_API_KEY      = os.getenv("POLYGON_API_KEY")

# Optional throttle: cap the universe size to bound memory/latency
UNIVERSE_MAX = int(os.getenv("UNIVERSE_MAX", "0"))


# --------------------------
# Sources / base candidates
# --------------------------
def _list_alpaca_assets() -> pd.DataFrame:
    """
    Return active, tradable US equities from Alpaca (symbol, name, exchange).
    Empty DataFrame on any auth/network error.
    """
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(
            key_id=APCA_API_KEY_ID,
            secret_key=APCA_API_SECRET_KEY,
            base_url=APCA_API_BASE_URL,
        )
        assets = api.list_assets(status='active')
        rows = []
        for a in assets:
            try:
                if getattr(a, 'tradable', False) and getattr(a, 'status', 'active') == 'active':
                    exch = getattr(a, 'exchange', '') or getattr(a, 'primary_exchange', '')
                    if exch in {'NYSE','NASDAQ','ARCA','BATS','AMEX'}:
                        cls = getattr(a, 'class', None) or getattr(a, 'asset_class', None)
                        if cls in {'us_equity','US_EQUITY', None}:
                            rows.append({'symbol': a.symbol, 'name': getattr(a, 'name', None), 'exchange': exch})
            except Exception:
                continue
        return pd.DataFrame(rows).drop_duplicates(subset=['symbol'])
    except Exception:
        return pd.DataFrame(columns=['symbol','name','exchange'])


def _seed_symbols_env() -> pd.DataFrame:
    """
    Optional bootstrap when Alpaca is unavailable or filtered out:
    SEED_SYMBOLS='AAPL,MSFT,SPY' -> DataFrame like _list_alpaca_assets().
    """
    syms = (os.getenv("SEED_SYMBOLS") or "").strip()
    if not syms:
        return pd.DataFrame(columns=['symbol','name','exchange'])
    items = [s.strip().upper() for s in syms.split(",") if s.strip()]
    return pd.DataFrame(
        [{'symbol': s, 'name': None, 'exchange': None} for s in items]
    ).drop_duplicates(subset=['symbol'])


# --------------------------
# Polygon enrichment (async)
# --------------------------
from utils_http import get_json_async

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

    infos = await asyncio.gather(*(_poly_ticker_info(s) for s in symbols))
    advs  = await asyncio.gather(*(_poly_adv(s, start, today) for s in symbols))

    rows = []
    for s, info, adv in zip(symbols, infos, advs):
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
    # No Polygon: keep shape but empty values
    return pd.DataFrame([{"symbol": s, "market_cap": None, "adv_usd_20": None} for s in symbols])


# --------------------------
# Rebuild universe (main)
# --------------------------
def _allowed_universe_columns() -> set[str]:
    """Columns that actually exist on the Universe table (for safe upserts)."""
    return {c.name for c in Universe.__table__.columns}

def rebuild_universe() -> pd.DataFrame:
    # 1) Base list from Alpaca; otherwise SEED_SYMBOLS
    base = _list_alpaca_assets()
    if base.empty:
        seed = _seed_symbols_env()
        if seed.empty:
            raise RuntimeError(
                "No assets from Alpaca and no SEED_SYMBOLS provided. "
                "Fix APCA_* credentials or set SEED_SYMBOLS to bootstrap."
            )
        base = seed

    # 2) Enrich with Polygon (market cap + ADV)
    enrich = _enrich_universe(base['symbol'].tolist())
    df = base.merge(enrich, on='symbol', how='left')

    # 3) Normalize types to avoid FutureWarnings / object math surprises
    for col in ("market_cap", "adv_usd_20"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4) Gates — tolerant on missing enrichment (pass-through)
    #    Tighten ADV_USD_MIN later once enrichment is flowing reliably.
    df["include_mc"]  = df.get("market_cap").isna()  | (df.get("market_cap") < MARKET_CAP_MAX)
    df["include_adv"] = df.get("adv_usd_20").isna() | (df.get("adv_usd_20") > ADV_USD_MIN)
    df["included"]    = df["include_mc"] & df["include_adv"]
    df["last_updated"] = pd.Timestamp.utcnow()

    # 5) Output columns (trim to table schema for safe upsert)
    out_cols_preferred = ["symbol","name","exchange","market_cap","adv_usd_20","included","last_updated"]
    allowed = _allowed_universe_columns()
    out_cols = [c for c in out_cols_preferred if c in allowed]

    out = (
        df.loc[df["included"], out_cols]
          .drop_duplicates(subset=["symbol"])
          .copy()
    )

    if out.empty:
        # Keep prior set rather than nuking on a bad day / rate limit blip
        raise RuntimeError("Universe filters produced empty set; keeping prior universe.")

    # 6) Optional cap to bound memory/latency
    if UNIVERSE_MAX > 0 and len(out) > UNIVERSE_MAX:
        # Prefer higher ADV if available, then smaller market cap
        if "adv_usd_20" in out.columns:
            out = out.sort_values("adv_usd_20", ascending=False)
        if "market_cap" in out.columns:
            out = out.sort_values("market_cap", ascending=True, kind="mergesort")
        out = out.head(UNIVERSE_MAX)

    # 7) Upsert rows; mark previously included-but-not-in-out as FALSE
    new_syms = tuple(out["symbol"].unique().tolist())

    with engine.begin() as con:
        # Use a conservative chunk size; helper will enforce param-limit anyway.
        upsert_dataframe(out, Universe, ["symbol"], chunk_size=5000, conn=con)
        if new_syms:
            stmt = (
                text("UPDATE universe SET included = FALSE WHERE included = TRUE AND symbol NOT IN :syms")
                .bindparams(bindparam("syms", expanding=True))
            )
            con.execute(stmt, {"syms": new_syms})
    return out


if __name__ == "__main__":
    rebuild_universe()
