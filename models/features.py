from __future__ import annotations
import numpy as np
import pandas as pd
from sqlalchemy import text, bindparam
from db import engine, upsert_dataframe, Feature
from typing import List

# Wilder's RSI using EWM alpha = 1/window
def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def _last_feature_dates() -> dict[str, pd.Timestamp]:
    df = pd.read_sql_query(text('SELECT symbol, MAX(ts) AS max_ts FROM features GROUP BY symbol'), engine, parse_dates=['max_ts'])
    return {r.symbol: r.max_ts for _, r in df.iterrows()}

def _symbols() -> list[str]:
    df = pd.read_sql_query(text('SELECT symbol FROM universe WHERE included = TRUE ORDER BY symbol'), engine)
    return df['symbol'].tolist()

def _load_prices_batch(symbols: List[str], start_ts: pd.Timestamp) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=['symbol','ts','close','adj_close','volume'])
    sql = (
        'SELECT symbol, ts, close, adj_close, volume '
        'FROM daily_bars '
        'WHERE ts >= :start '
        '  AND symbol IN :syms '
        'ORDER BY symbol, ts'
    )
    stmt = text(sql).bindparams(bindparam('syms', expanding=True))
    return pd.read_sql_query(stmt, engine, params={'start': start_ts.date(), 'syms': tuple(symbols)}, parse_dates=['ts'])

def _load_fundamentals(symbols: List[str]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=['symbol','as_of'])
    sql = (
        'SELECT symbol, as_of, pe_ttm, pb, ps_ttm, debt_to_equity, return_on_assets, gross_margins, profit_margins, current_ratio '
        'FROM fundamentals '
        'WHERE symbol IN :syms '
        'ORDER BY symbol, as_of'
    )
    stmt = text(sql).bindparams(bindparam('syms', expanding=True))
    return pd.read_sql_query(stmt, engine, params={'syms': tuple(symbols)}, parse_dates=['as_of'])

def build_features(batch_size: int = 200, warmup_days: int = 90) -> pd.DataFrame:
    syms = _symbols()
    if not syms:
        return pd.DataFrame()

    last_map = _last_feature_dates()
    uni = pd.read_sql_query(text('SELECT symbol, market_cap FROM universe WHERE included = TRUE'), engine).set_index('symbol')

    all_new_rows: list[pd.DataFrame] = []

    for i in range(0, len(syms), batch_size):
        bsyms = syms[i:i+batch_size]
        starts = []
        for s in bsyms:
            last_ts = last_map.get(s)
            if pd.isna(last_ts) or last_ts is None:
                starts.append(pd.Timestamp('1900-01-01'))
            else:
                starts.append(pd.Timestamp(last_ts) - pd.Timedelta(days=warmup_days))
        start_ts = min(starts) if starts else pd.Timestamp('1900-01-01')

        px = _load_prices_batch(bsyms, start_ts)
        if px.empty:
            continue
        fnd = _load_fundamentals(bsyms)

        px['price_feat'] = px['adj_close'].where(px['adj_close'].notna(), px['close'])

        out_frames: list[pd.DataFrame] = []
        for sym, g in px.groupby('symbol'):
            g = g.sort_values('ts').copy()
            p = g['price_feat']

            # Essential time-series calcs
            g['ret_1d'] = p.pct_change(1)
            g['ret_5d'] = p.pct_change(5)
            g['ret_21d'] = p.pct_change(21)
            g['mom_21'] = (p / p.shift(21)) - 1.0
            g['mom_63'] = (p / p.shift(63)) - 1.0
            g['vol_21'] = g['ret_1d'].rolling(21).std()
            g['rsi_14'] = _compute_rsi(p, 14)

            # Adjusted turnover uses price_feat
            dv = (g['price_feat'] * g['volume']).rolling(21).mean()
            mc = float(uni.loc[sym, 'market_cap']) if sym in uni.index and pd.notnull(uni.loc[sym, 'market_cap']) else np.nan
            g['turnover_21'] = (dv / mc) if (mc and mc > 0) else np.nan
            g['size_ln'] = np.log(mc) if (mc and mc > 0) else np.nan

            # Merge fundamentals point-in-time (backward as-of)
            f_sym = fnd[fnd['symbol'] == sym][['as_of','pe_ttm','pb','ps_ttm','debt_to_equity','return_on_assets','gross_margins','profit_margins','current_ratio']].sort_values('as_of')
            if not f_sym.empty:
                g = pd.merge_asof(
                    g.sort_values('ts'),
                    f_sym.rename(columns={'as_of':'ts_fnd'}).sort_values('ts_fnd'),
                    left_on='ts',
                    right_on='ts_fnd',
                    direction='backward'
                )
            else:
                g['pe_ttm'] = np.nan
                g['pb'] = np.nan
                g['ps_ttm'] = np.nan
                g['debt_to_equity'] = np.nan
                g['return_on_assets'] = np.nan
                g['gross_margins'] = np.nan
                g['profit_margins'] = np.nan
                g['current_ratio'] = np.nan

            # Feature columns
            g['f_pe_ttm'] = g['pe_ttm']
            g['f_pb'] = g['pb']
            g['f_ps_ttm'] = g['ps_ttm']
            g['f_debt_to_equity'] = g['debt_to_equity']
            g['f_roa'] = g['return_on_assets']
            g['f_gm'] = g['gross_margins']
            g['f_profit_margin'] = g['profit_margins']
            g['f_current_ratio'] = g['current_ratio']

            last_ts = last_map.get(sym)
            if last_ts is not None and not pd.isna(last_ts):
                g = g[g['ts'] > pd.Timestamp(last_ts)]

            fcols = [
                'symbol','ts','ret_1d','ret_5d','ret_21d','mom_21','mom_63','vol_21','rsi_14','turnover_21','size_ln',
                'f_pe_ttm','f_pb','f_ps_ttm','f_debt_to_equity','f_roa','f_gm','f_profit_margin','f_current_ratio'
            ]

            # Drop only if essential TS calcs missing
            essentials = ['ret_1d','ret_5d','ret_21d','mom_21','mom_63','vol_21','rsi_14']
            g = g.dropna(subset=essentials, how='any')
            out_frames.append(g[fcols])

        if out_frames:
            feats = pd.concat(out_frames, ignore_index=True)
            upsert_dataframe(feats, Feature, ['symbol','ts'])
            all_new_rows.append(feats)

    return pd.concat(all_new_rows, ignore_index=True) if all_new_rows else pd.DataFrame()
