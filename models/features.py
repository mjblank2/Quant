from __future__ import annotations
import numpy as np
import pandas as pd
from sqlalchemy import text, bindparam
from db import engine, upsert_dataframe, Feature
from config import ADV_LOOKBACK

FEATURE_COLS = ["ret_1d","ret_5d","ret_21d","mom_21","mom_63","vol_21","rsi_14","turnover_21","size_ln"]

def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(window).mean()
    roll_down = pd.Series(down, index=series.index).rolling(window).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _last_feature_dates() -> dict[str, pd.Timestamp]:
    df = pd.read_sql_query(text("SELECT symbol, MAX(ts) AS max_ts FROM features GROUP BY symbol"), engine, parse_dates=["max_ts"])    
    return {r.symbol: r.max_ts for _, r in df.iterrows()}

def _symbols() -> list[str]:
    df = pd.read_sql_query(text("SELECT symbol FROM universe WHERE included = TRUE ORDER BY symbol"), engine)
    return df['symbol'].tolist()

def _load_prices_batch(symbols: list[str], start_ts: pd.Timestamp) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["symbol","ts","close","adj_close","volume"])
    stmt = text("""
        SELECT symbol, ts, close, adj_close, volume
        FROM daily_bars
        WHERE ts >= :start
          AND symbol IN :syms
        ORDER BY symbol, ts
    """).bindparams(bindparam("syms", expanding=True))
    return pd.read_sql_query(stmt, engine, params={"start": start_ts.date(), "syms": tuple(symbols)}, parse_dates=["ts"])

def build_features(batch_size: int = 200, warmup_days: int = 90) -> pd.DataFrame:
    syms = _symbols()
    if not syms:
        return pd.DataFrame()

    last_map = _last_feature_dates()
    # Universe snapshot for size (market cap)
    uni = pd.read_sql_query(text("SELECT symbol, market_cap FROM universe WHERE included = TRUE"), engine).set_index("symbol")

    all_new_rows = []

    for i in range(0, len(syms), batch_size):
        bsyms = syms[i:i+batch_size]
        # Earliest start among this batch = min(last_ts - warmup) for known symbols, else very early
        starts = []
        for s in bsyms:
            last_ts = last_map.get(s)
            if pd.isna(last_ts) or last_ts is None:
                starts.append(pd.Timestamp("1900-01-01"))
            else:
                starts.append(pd.Timestamp(last_ts) - pd.Timedelta(days=warmup_days))
        start_ts = min(starts) if starts else pd.Timestamp("1900-01-01")

        px = _load_prices_batch(bsyms, start_ts)
        if px.empty:
            continue

        # Prefer adjusted close for returns/momentum; fallback to close
        px["price_feat"] = px["adj_close"].where(px["adj_close"].notna(), px["close"])

        out_frames = []
        for sym, g in px.groupby("symbol"):
            g = g.sort_values("ts").copy()
            p = g["price_feat"]
            g["ret_1d"] = p.pct_change(1)
            g["ret_5d"] = p.pct_change(5)
            g["ret_21d"] = p.pct_change(21)
            g["mom_21"] = (p / p.shift(21)) - 1.0
            g["mom_63"] = (p / p.shift(63)) - 1.0
            g["vol_21"] = g["ret_1d"].rolling(21).std()
            g["rsi_14"] = _compute_rsi(p, 14)

            # Dollar volume for turnover (use raw close)
            dv = (g["close"] * g["volume"]).rolling(21).mean()
            mc = float(uni.loc[sym, "market_cap"]) if sym in uni.index and pd.notnull(uni.loc[sym, "market_cap"]) else np.nan
            g["turnover_21"] = (dv / mc) if mc and mc > 0 else np.nan
            g["size_ln"] = np.log(mc) if mc and mc > 0 else np.nan

            # filter to only rows strictly newer than last feature date
            last_ts = last_map.get(sym)
            if last_ts is not None and not pd.isna(last_ts):
                g = g[g["ts"] > pd.Timestamp(last_ts)]
            # Keep feature columns
            fcols = ["symbol","ts","ret_1d","ret_5d","ret_21d","mom_21","mom_63","vol_21","rsi_14","turnover_21","size_ln"]
            out_frames.append(g[fcols].dropna())

        if out_frames:
            feats = pd.concat(out_frames, ignore_index=True)
            upsert_dataframe(feats, Feature, ["symbol","ts"])
            all_new_rows.append(feats)

    return pd.concat(all_new_rows, ignore_index=True) if all_new_rows else pd.DataFrame()
