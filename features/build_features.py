
from __future__ import annotations

import logging
from typing import List
import numpy as np
import pandas as pd
from sqlalchemy import text, bindparam

from db import engine, upsert_dataframe, Feature  # type: ignore
from config import TARGET_HORIZON_DAYS
from utils.price_utils import price_expr, select_price_as

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
    """
    Load raw price data for a batch of symbols, including open, high, low, close,
    adjusted close, and volume.  The high/low fields are needed for ATR and
    spread calculations.
    """
    if not symbols:
        # return DataFrame with all expected columns when no symbols are provided
        return pd.DataFrame(columns=['symbol','ts','open','high','low','close','adj_close','volume'])

    sql = f"""
        SELECT symbol, ts, open, high, low, close,
               {select_price_as('adj_close')}, volume
        FROM daily_bars
        WHERE ts >= :start AND symbol IN :syms
        ORDER BY symbol, ts
    """
    stmt = text(sql).bindparams(bindparam('syms', expanding=True))
    params = {'start': start_ts.date(), 'syms': tuple(symbols)}
    return pd.read_sql_query(stmt, engine, params=params, parse_dates=['ts'])

def _load_market_returns(market_symbol: str = "IWM") -> pd.DataFrame:
    """Return market prices and daily returns for the benchmark symbol."""
    try:
        df = pd.read_sql_query(
            text(f"SELECT ts, {select_price_as('px')} FROM daily_bars WHERE symbol=:m ORDER BY ts"),
            engine,
            params={"m": market_symbol},
            parse_dates=["ts"],
        )
        if df.empty:
            return pd.DataFrame(columns=["ts", "px", "mret"])
        df["mret"] = df["px"].pct_change()
        return df[["ts", "px", "mret"]]
    except Exception:
        return pd.DataFrame(columns=["ts", "px", "mret"])

def _load_fundamentals(symbols: List[str]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=['symbol','as_of'])
    sql = (
        'SELECT symbol, as_of, pe_ttm, pb, ps_ttm, debt_to_equity, return_on_assets, '
        'gross_margins, profit_margins, current_ratio '
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

def _load_alt_signals(symbols: List[str], start_ts: pd.Timestamp) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=['symbol','ts'])
    sql = (
        'SELECT symbol, ts, name, value FROM alt_signals '
        'WHERE ts >= :start AND symbol IN :syms ORDER BY symbol, ts'
    )
    stmt = text(sql).bindparams(bindparam('syms', expanding=True))
    df = pd.read_sql_query(stmt, engine, params={'start': start_ts.date(), 'syms': tuple(symbols)}, parse_dates=['ts'])
    if df.empty:
        return pd.DataFrame(columns=['symbol','ts'])
    piv = df.pivot_table(index=['symbol','ts'], columns='name', values='value', aggfunc='last').reset_index()
    return piv

def build_features(batch_size: int = 200, warmup_days: int = 90) -> pd.DataFrame:
    """
    Incremental point‑in‑time feature builder.  Retrieves price, fundamental,
    shares‑outstanding and alternative signal data; computes a broad set of
    technical indicators (returns, momentum, volatility, RSI, MACD, long‑term
    momentum/volatility, spread ratios, ATR, OBV, etc.), as well as microstructure
    features and fundamental signals; and persists them to the database using
    an upsert pattern.
    """
    log.info("Starting feature build process (Incremental, PIT).")
    syms = _symbols()
    if not syms:
        log.warning("Universe is empty. Cannot build features.")
        return pd.DataFrame()

    last_map = _last_feature_dates()
    new_rows: list[pd.DataFrame] = []
    # Load market data once for beta calculations
    mkt = _load_market_returns("IWM")
    if not mkt.empty:
        mkt = mkt.rename(columns={"px":"mkt_px", "mret":"mkt_ret"})

    # Process symbols in batches to control memory usage
    for i in range(0, len(syms), batch_size):
        bsyms = syms[i:i+batch_size]
        log.info(f"Processing batch {i//batch_size + 1} / {int(np.ceil(len(syms)/batch_size))} "
                 f"(Symbols: {len(bsyms)})")

        # Determine earliest timestamp to load for each symbol (for warmup)
        starts = []
        for s in bsyms:
            last_ts = last_map.get(s)
            if last_ts is None or pd.isna(last_ts):
                starts.append(pd.Timestamp("1900-01-01"))
            else:
                starts.append(pd.Timestamp(last_ts) - pd.Timedelta(days=warmup_days))
        start_ts = min(starts) if starts else pd.Timestamp("1900-01-01")

        # Load price data, fundamentals, shares outstanding, alt signals
        px = _load_prices_batch(bsyms, start_ts)
        if px.empty:
            continue

        fnd = _load_fundamentals(bsyms)
        shs = _load_shares_outstanding(bsyms)
        alts = _load_alt_signals(bsyms, start_ts)

        # Merge alternative signals into price data; fill missing alt columns with zeros
        if not alts.empty:
            px = px.merge(alts, on=['symbol','ts'], how='left')
        for c in ['pead_event','pead_surprise_eps','pead_surprise_rev','russell_inout']:
            if c not in px.columns:
                px[c] = 0.0
            else:
                px[c] = px[c].fillna(0.0)

        # Choose adjusted close when available, else close
        px['price_feat'] = px['adj_close'].where(px['adj_close'].notna(), px['close'])

        # Merge market returns (for beta and residual calculations)
        if not mkt.empty:
            px = px.merge(mkt, on='ts', how='left')

        out_frames: list[pd.DataFrame] = []

        # Compute features per symbol
        for sym, g in px.groupby('symbol'):
            g = g.sort_values('ts').copy()
            p = g['price_feat']  # price series for technical calculations

            # Price/momentum/volatility features
            g['ret_1d'] = p.pct_change(1)
            g['ret_5d'] = p.pct_change(5)
            g['ret_21d'] = p.pct_change(21)
            g['mom_21'] = (p / p.shift(21)) - 1.0
            g['mom_63'] = (p / p.shift(63)) - 1.0
            g['vol_21'] = g['ret_1d'].rolling(21).std()
            g['rsi_14'] = _compute_rsi(p, 14)

            # Volatility‑adjusted short‑term reversal (5‑day)
            lagged_vol = g['vol_21'].shift(1)
            denom = (lagged_vol * np.sqrt(5)).replace(0, np.nan)
            g['reversal_5d_z'] = -(g['ret_5d'] / denom)

            # Microstructure: overnight gap, average daily dollar volume & Amihud illiquidity
            g['overnight_gap'] = (g['open'] / g['price_feat'].shift(1)) - 1.0
            dollar_volume = g['price_feat'] * g['volume']
            g['adv_usd_21'] = dollar_volume.rolling(21).mean()
            g['illiq_21'] = (g['ret_1d'].abs() / dollar_volume.replace(0,np.nan)).rolling(21).mean()

            # **New technical indicators**

            # Exponential moving averages & MACD
            g['ema_12'] = p.ewm(span=12, adjust=False).mean()
            g['ema_26'] = p.ewm(span=26, adjust=False).mean()
            g['macd'] = g['ema_12'] - g['ema_26']
            g['ema_50'] = p.ewm(span=50, adjust=False).mean()
            g['ema_200'] = p.ewm(span=200, adjust=False).mean()
            g['ma_ratio_50_200'] = g['ema_50'] / g['ema_200']

            # Long‑term volatility and momentum
            g['vol_63'] = g['ret_1d'].rolling(63).std()
            g['vol_252'] = g['ret_1d'].rolling(252).std()
            g['mom_252'] = (p / p.shift(252)) - 1.0

            # Intraday spread ratio & its 21‑day average
            g['spread_ratio'] = (g['high'] - g['low']) / g['price_feat']
            g['spread_21'] = g['spread_ratio'].rolling(21).mean()

            # Average True Range (ATR) over 14 days
            tr = pd.concat([
                g['high'] - g['low'],
                (g['high'] - g['price_feat'].shift(1)).abs(),
                (g['low'] - g['price_feat'].shift(1)).abs()
            ], axis=1).max(axis=1)
            g['atr_14'] = tr.rolling(14).mean()

            # On‑Balance Volume (OBV)
            direction = np.where(g['ret_1d'] >= 0, 1, -1)
            g['obv'] = (direction * g['volume']).fillna(0).cumsum()

            # Shares outstanding → market cap & turnover
            shs_sym = shs[shs['symbol'] == sym][['as_of','shares']].sort_values('as_of')
            if not shs_sym.empty:
                g = pd.merge_asof(
                    g.sort_values('ts'),
                    shs_sym.rename(columns={'as_of':'ts_shs'}).sort_values('ts_shs'),
                    left_on='ts', right_on='ts_shs', direction='backward'
                )
                g['market_cap_pit'] = g['price_feat'] * g['shares']
            else:
                # fallback: estimate shares from median volume (very conservative)
                median_volume = g['volume'].median()
                estimated_shares = median_volume * 10
                g['market_cap_pit'] = g['price_feat'] * estimated_shares

            mc = g['market_cap_pit']
            g['size_ln'] = np.log(mc.clip(lower=1.0))
            g['turnover_21'] = g['adv_usd_21'] / mc.replace(0, np.nan)

            # Point‑in‑time fundamentals (merge as of most recent available date)
            f_sym = fnd[fnd['symbol'] == sym].drop(columns=['symbol']).sort_values('as_of')
            if not f_sym.empty:
                g = pd.merge_asof(
                    g.sort_values('ts'),
                    f_sym.rename(columns={'as_of':'ts_fnd'}).sort_values('ts_fnd'),
                    left_on='ts', right_on='ts_fnd', direction='backward'
                )
            # Ensure fundamental columns exist
            for col in ['pe_ttm','pb','ps_ttm','debt_to_equity','return_on_assets','gross_margins','profit_margins','current_ratio']:
                if col not in g.columns:
                    g[col] = np.nan

            # Rolling beta vs. market
            if 'mkt_ret' in g.columns and not g['mkt_ret'].isna().all():
                cov = g['ret_1d'].rolling(63).cov(g['mkt_ret'])
                var = g['mkt_ret'].rolling(63).var()
                g['beta_63'] = cov / var.replace(0, np.nan)
            else:
                g['beta_63'] = 1.0

            # Idiosyncratic volatility
            if 'mkt_ret' in g.columns and not g['mkt_ret'].isna().all():
                resid = g['ret_1d'] - g['beta_63'] * g['mkt_ret']
                g['ivol_63'] = resid.rolling(63).std()
            else:
                g['ivol_63'] = g['ret_1d'].rolling(63).std()

            # Forward returns & residual returns
            g['px_fwd'] = g['price_feat'].shift(-TARGET_HORIZON_DAYS)
            g['fwd_ret'] = (g['px_fwd'] / g['price_feat']) - 1.0
            if 'mkt_px' in g.columns:
                g['mkt_px_fwd'] = g['mkt_px'].shift(-TARGET_HORIZON_DAYS)
                g['mkt_fwd_ret'] = (g['mkt_px_fwd'] / g['mkt_px']) - 1.0
                g['fwd_ret_resid'] = g['fwd_ret'] - g['beta_63'].fillna(1.0) * g['mkt_fwd_ret'].fillna(0.0)
            else:
                g['fwd_ret_resid'] = g['fwd_ret']

            # Filter out already‑processed timestamps for incremental build
            last_ts = last_map.get(sym)
            if last_ts is not None and not pd.isna(last_ts):
                g = g[g['ts'] > pd.Timestamp(last_ts)]

            # Column list for upsert
            fcols = [
                'symbol','ts','ret_1d','ret_5d','ret_21d','mom_21','mom_63','vol_21','rsi_14',
                'reversal_5d_z','ivol_63','turnover_21','size_ln','adv_usd_21','overnight_gap',
                'illiq_21','beta_63','fwd_ret','fwd_ret_resid',
                'pead_event','pead_surprise_eps','pead_surprise_rev','russell_inout',
                # fundamental columns mapped below
                'f_pe_ttm','f_pb','f_ps_ttm','f_debt_to_equity','f_roa','f_gm','f_profit_margin','f_current_ratio',
                # new technical features
                'ema_12','ema_26','macd','ema_50','ema_200','ma_ratio_50_200',
                'vol_63','vol_252','mom_252','spread_ratio','spread_21','atr_14','obv'
            ]

            # Map PIT fundamental fields to unified names
            g['f_pe_ttm'] = g.get('pe_ttm')
            g['f_pb'] = g.get('pb')
            g['f_ps_ttm'] = g.get('ps_ttm')
            g['f_debt_to_equity'] = g.get('debt_to_equity')
            g['f_roa'] = g.get('return_on_assets')
            g['f_gm'] = g.get('gross_margins')
            g['f_profit_margin'] = g.get('profit_margins')
            g['f_current_ratio'] = g.get('current_ratio')

            # Drop rows with missing core features to avoid NaNs in training
            core_features = ['ret_1d','ret_5d','vol_21']
            g2 = g.dropna(subset=core_features)[fcols].copy()
            if len(g2) > 0:
                g2 = g2.drop_duplicates(subset=['symbol','ts'], keep='last')
                out_frames.append(g2)

        # Upsert new rows for this batch
        if out_frames:
            feats = pd.concat(out_frames, ignore_index=True)
            feats = feats.sort_values('ts').drop_duplicates(['symbol','ts'], keep='last').reset_index(drop=True)
            upsert_dataframe(feats, Feature, ['symbol','ts'], chunk_size=200)
            new_rows.append(feats)
            log.info(f"Batch completed. New rows upserted: {len(feats)}")

    return pd.concat(new_rows, ignore_index=True) if new_rows else pd.DataFrame()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_features()

