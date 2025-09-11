from __future__ import annotations
import numpy as np
import pandas as pd
from sqlalchemy import text, bindparam
from typing import List
from db import engine, upsert_dataframe, Feature
from utils.price_utils import price_expr
import logging

log = logging.getLogger(__name__)

def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Computes Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    # Add epsilon to avoid division by zero
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

    # Use dynamic price expression based on column availability
    sql = (
        f'SELECT symbol, ts, open, close, {price_expr()} as adj_close, volume '
        'FROM daily_bars '
        'WHERE ts >= :start AND symbol IN :syms '
        'ORDER BY symbol, ts'
    )

    stmt = text(sql).bindparams(bindparam('syms', expanding=True))
    params = {'start': start_ts.date(), 'syms': tuple(symbols)}

    try:
        return pd.read_sql_query(stmt, engine, params=params, parse_dates=['ts'])
    except Exception as e:
        log.error(f"Failed to load price data: {e}")
        return pd.DataFrame(columns=['symbol', 'ts', 'open', 'close', 'adj_close', 'volume'])

def _load_market_returns(market_symbol: str = "IWM") -> pd.DataFrame:
    try:
        df = pd.read_sql_query(text(f"""
            SELECT ts, {price_expr()} AS px
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

def build_features(batch_size: int = 200, warmup_days: int = 90) -> pd.DataFrame:
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
        out_frames: list[pd.DataFrame] = []

        # Pre-merge market returns for beta/idio-vol computations
        if not mkt.empty:
            px = px.merge(mkt, on='ts', how='left')

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
                # Fallback: estimate market cap using median volume-to-shares ratio
                median_volume = g['volume'].median()
                estimated_shares = median_volume * 10  # Conservative estimate
                g['market_cap_pit'] = g['price_feat'] * estimated_shares
                log.debug(f"No shares outstanding for {sym}, using estimated market cap")

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

            # Rolling beta vs IWM (if available)
            if 'mret' in g.columns and not g['mret'].isna().all():
                cov = g['ret_1d'].rolling(63).cov(g['mret'])
                var = g['mret'].rolling(63).var()
                g['beta_63'] = cov / var.replace(0, np.nan)
            else:
                g['beta_63'] = 1.0 # Fallback
                if 'mret' not in g.columns:
                    log.debug(f"No market returns data, using beta=1.0 for {sym}")

            # finalize
            last_ts = last_map.get(sym)
            if last_ts is not None and not pd.isna(last_ts):
                g = g[g['ts'] > pd.Timestamp(last_ts)]

            fcols = ['symbol','ts'] + ['ret_1d','ret_5d','ret_21d','mom_21','mom_63','vol_21','rsi_14','turnover_21','size_ln','adv_usd_21','overnight_gap','illiq_21','beta_63'] + ['f_pe_ttm','f_pb','f_ps_ttm','f_debt_to_equity','f_roa','f_gm','f_profit_margin','f_current_ratio']
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
                out_frames.append(g2)

        if out_frames:
            feats = pd.concat(out_frames, ignore_index=True)
            feats = (
                feats.sort_values('ts')
                .drop_duplicates(['symbol', 'ts'], keep='last')
                .reset_index(drop=True)
            )
            upsert_dataframe(feats, Feature, ['symbol', 'ts'], chunk_size=200)
            new_rows.append(feats)
            log.info(f"Batch completed. New rows: {len(feats)}")

    return pd.concat(new_rows, ignore_index=True) if new_rows else pd.DataFrame()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_features()

