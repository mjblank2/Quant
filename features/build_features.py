
from __future__ import annotations

import logging
from typing import List
import numpy as np
import pandas as pd
from sqlalchemy import text, bindparam
import gc

from db import engine, upsert_dataframe, Feature  # type: ignore

log = logging.getLogger(__name__)

def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs))

def _last_feature_dates() -> dict[str, pd.Timestamp]:
    try:
        df = pd.read_sql_query(text('SELECT symbol, MAX(ts) AS max_ts FROM features GROUP BY symbol'), engine, parse_dates=['max_ts'])
        return {r.symbol: r.max_ts for _, r in df.iterrows()}
    except Exception:
        return {}

def _symbols() -> list[str]:
    try:
        df = pd.read_sql_query(text('SELECT symbol FROM universe WHERE included = TRUE ORDER BY symbol'), engine)
        return df['symbol'].tolist()
    except Exception:
        return []

def _load_prices_batch(symbols: List[str], start_ts: pd.Timestamp) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=['symbol', 'ts', 'open', 'close', 'adj_close', 'volume'])

    sql_with_adj = (
        'SELECT symbol, ts, open, close, COALESCE(adj_close, close) as adj_close, volume '
        'FROM daily_bars '
        'WHERE ts >= :start AND symbol IN :syms '
        'ORDER BY symbol, ts'
    )
    sql_fallback = (
        'SELECT symbol, ts, open, close, close as adj_close, volume '
        'FROM daily_bars '
        'WHERE ts >= :start AND symbol IN :syms '
        'ORDER BY symbol, ts'
    )
    stmt_with_adj = text(sql_with_adj).bindparams(bindparam('syms', expanding=True))
    stmt_fallback = text(sql_fallback).bindparams(bindparam('syms', expanding=True))
    params = {'start': start_ts.date(), 'syms': tuple(symbols)}
    try:
        return pd.read_sql_query(stmt_with_adj, engine, params=params, parse_dates=['ts'])
    except Exception as e:
        if "adj_close" in str(e) and ("no such column" in str(e) or "does not exist" in str(e)):
            log.warning("adj_close column not found in daily_bars table, using close price instead")
            return pd.read_sql_query(stmt_fallback, engine, params=params, parse_dates=['ts'])
        else:
            raise

def _load_market_returns(market_symbol: str = "IWM") -> pd.DataFrame:
    try:
        df = pd.read_sql_query(text("""
            SELECT ts, COALESCE(adj_close, close) AS px
            FROM daily_bars WHERE symbol=:m ORDER BY ts
        """), engine, params={'m': market_symbol}, parse_dates=['ts'])
        if df.empty:
            return pd.DataFrame(columns=['ts','mret'])
        df['mret'] = df['px'].pct_change()
        return df[['ts','mret']]
    except Exception:
        return pd.DataFrame(columns=['ts','mret'])

def _load_fundamentals(symbols: List[str]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=['symbol','as_of'])
    sql = (
        'SELECT symbol, as_of, pe_ttm, pb, ps_ttm, debt_to_equity, return_on_assets, gross_margins, profit_margins, current_ratio '
        'FROM fundamentals WHERE symbol IN :syms ORDER BY symbol, as_of'
    )
    stmt = text(sql).bindparams(bindparam('syms', expanding=True))
    return pd.read_sql_query(stmt, engine, params={'syms': tuple(symbols)}, parse_dates=['as_of'])

def _load_shares_outstanding(symbols: List[str]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=['symbol','as_of','shares'])
    sql = 'SELECT symbol, as_of, shares FROM shares_outstanding WHERE symbol IN :syms ORDER BY symbol, as_of'
    stmt = text(sql).bindparams(bindparam('syms', expanding=True))
    return pd.read_sql_query(stmt, engine, params={'syms': tuple(symbols)}, parse_dates=['as_of'])

def build_features(batch_size: int = 100, warmup_days: int = 90) -> pd.DataFrame:
    """Incremental, point-in-time feature building with duplicate-safe upsert."""
    log.info("Starting feature build process (Incremental, PIT).")
    syms = _symbols()
    if not syms:
        log.warning("Universe is empty. Cannot build features.")
        return pd.DataFrame()

    last_map = _last_feature_dates()
    new_rows: list[pd.DataFrame] = []
    mkt = _load_market_returns("IWM")

    for i in range(0, len(syms), batch_size):
        bsyms = syms[i:i+batch_size]
        log.info(f"Processing batch {i//batch_size + 1} / {int(np.ceil(len(syms)/batch_size))} (Symbols: {len(bsyms)})")

        starts = []
        for s in bsyms:
            last_ts = last_map.get(s)
            starts.append(pd.Timestamp('1900-01-01') if (last_ts is None or pd.isna(last_ts)) else pd.Timestamp(last_ts) - pd.Timedelta(days=warmup_days))
        start_ts = min(starts) if starts else pd.Timestamp('1900-01-01')

        px = _load_prices_batch(bsyms, start_ts)
        if px.empty:
            continue

        fnd = _load_fundamentals(bsyms)
        shs = _load_shares_outstanding(bsyms)

        px['price_feat'] = px['adj_close'].where(px['adj_close'].notna(), px['close'])

        # Merge market returns once for beta calculations
        if not mkt.empty:
            px = px.merge(mkt, on='ts', how='left')

        out_frames: list[pd.DataFrame] = []

        for sym, g in px.groupby('symbol'):
            g = g.sort_values('ts').copy()
            p = g['price_feat']

            # Price/Momentum/Vol
            g['ret_1d'] = p.pct_change(1)
            g['ret_5d'] = p.pct_change(5)
            g['ret_21d'] = p.pct_change(21)
            g['mom_21'] = (p / p.shift(21)) - 1.0
            g['mom_63'] = (p / p.shift(63)) - 1.0
            g['vol_21'] = g['ret_1d'].rolling(21).std()
            g['rsi_14'] = _compute_rsi(p, 14)

            # Robust short-term reversal (vol-adjusted) on 5-day returns
            # Lagged volatility is used to avoid look-ahead bias.  Multiply by sqrt(5)
            # so that the denominator is the volatility of a 5-day sum under independence.
            lagged_vol = g['vol_21'].shift(1)
            denom = (lagged_vol * np.sqrt(5)).replace(0, np.nan)
            g['reversal_5d_z'] = -(g['ret_5d'] / denom)

            # Microstructure: overnight gap and Amihud illiquidity
            g['overnight_gap'] = (g['open'] / g['price_feat'].shift(1)) - 1.0
            dollar_volume = g['price_feat'] * g['volume']
            g['adv_usd_21'] = dollar_volume.rolling(21).mean()
            g['illiq_21'] = (g['ret_1d'].abs() / dollar_volume.replace(0, np.nan)).rolling(21).mean()

            # PIT Shares → Market Cap
            shs_sym = shs[shs['symbol'] == sym][['as_of','shares']].sort_values('as_of')
            if not shs_sym.empty:
                g = pd.merge_asof(
                    g.sort_values('ts'),
                    shs_sym.rename(columns={'as_of':'ts_shs'}).sort_values('ts_shs'),
                    left_on='ts', right_on='ts_shs', direction='backward'
                )
                g['market_cap_pit'] = g['price_feat'] * g['shares']
            else:
                median_volume = g['volume'].median()
                estimated_shares = median_volume * 10  # Conservative estimate (fallback)
                g['market_cap_pit'] = g['price_feat'] * estimated_shares

            mc = g['market_cap_pit']
            g['size_ln'] = np.log(mc.clip(lower=1.0))
            g['turnover_21'] = g['adv_usd_21'] / mc.replace(0, np.nan)

            # PIT Fundamentals
            f_sym = fnd[fnd['symbol'] == sym].drop(columns=['symbol']).sort_values('as_of')
            if not f_sym.empty:
                g = pd.merge_asof(
                    g.sort_values('ts'),
                    f_sym.rename(columns={'as_of':'ts_fnd'}).sort_values('ts_fnd'),
                    left_on='ts', right_on='ts_fnd', direction='backward'
                )
            for col in ['pe_ttm','pb','ps_ttm','debt_to_equity','return_on_assets','gross_margins','profit_margins','current_ratio']:
                if col not in g.columns:
                    g[col] = np.nan

            # Rolling beta vs market (if available)
            if 'mret' in g.columns and not g['mret'].isna().all():
                cov = g['ret_1d'].rolling(63).cov(g['mret'])
                var = g['mret'].rolling(63).var()
                g['beta_63'] = cov / var.replace(0, np.nan)
            else:
                g['beta_63'] = 1.0

            last_ts = last_map.get(sym)
            if last_ts is not None and not pd.isna(last_ts):
                g = g[g['ts'] > pd.Timestamp(last_ts)]

            fcols = ['symbol','ts','ret_1d','ret_5d','ret_21d','mom_21','mom_63','vol_21','rsi_14',
                     'turnover_21','size_ln','adv_usd_21','overnight_gap','illiq_21','beta_63',
                     'f_pe_ttm','f_pb','f_ps_ttm','f_debt_to_equity','f_roa','f_gm','f_profit_margin','f_current_ratio']

            # map PIT fund columns to unified names used by downstream model
            g['f_pe_ttm'] = g.get('pe_ttm')
            g['f_pb'] = g.get('pb')
            g['f_ps_ttm'] = g.get('ps_ttm')
            g['f_debt_to_equity'] = g.get('debt_to_equity')
            g['f_roa'] = g.get('return_on_assets')
            g['f_gm'] = g.get('gross_margins')
            g['f_profit_margin'] = g.get('profit_margins')
            g['f_current_ratio'] = g.get('current_ratio')

            core_features = ['ret_1d', 'ret_5d', 'vol_21']
            g2 = g.dropna(subset=core_features)[fcols].copy()
            if len(g2) > 0:
                # Defensive: remove duplicate keys to avoid ON CONFLICT hitting same row twice
                g2 = g2.drop_duplicates(subset=['symbol','ts'], keep='last')
                out_frames.append(g2)

        if out_frames:
            feats = pd.concat(out_frames, ignore_index=True)
            # Extra defensive de-dupe at batch level
            feats = feats.sort_values(['symbol','ts']).drop_duplicates(subset=['symbol','ts'], keep='last')
            upsert_dataframe(feats, Feature, ['symbol','ts'])
            new_rows.append(feats)
            log.info(f"Batch completed. New rows upserted: {len(feats)}")

        # Explicitly free large temporary objects to keep memory usage in check
        try:
            del px, fnd, shs, out_frames
        except NameError:
            pass
        gc.collect()

    return pd.concat(new_rows, ignore_index=True) if new_rows else pd.DataFrame()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_features()
