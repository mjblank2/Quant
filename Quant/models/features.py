"""
Feature engineering for small‑cap quantitative models.

This module constructs a rich set of technical, microstructure and
fundamental features from raw daily price data, market returns and
point‑in‑time fundamentals.  The resulting features feed into
machine‑learning models used to predict future returns.  In addition to
existing fundamental ratios (PE, PB, PS, debt‑to‑equity, return on assets,
gross and profit margins, and current ratio), this version introduces
**return on equity (ROE)** as an additional factor.  ROE captures how
efficiently a company generates profits relative to shareholder equity and
is commonly used in value and quality models.
"""

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
    """Compute the Relative Strength Index (RSI) over a rolling window."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs))


def _last_feature_dates() -> dict[str, pd.Timestamp]:
    """Return a mapping from symbol to the most recent feature timestamp."""
    try:
        df = pd.read_sql_query(text('SELECT symbol, MAX(ts) AS max_ts FROM features GROUP BY symbol'), engine, parse_dates=['max_ts'])
        return {r.symbol: r.max_ts for _, r in df.iterrows()}
    except Exception:
        return {}


def _symbols() -> list[str]:
    """Return the list of included symbols in the universe."""
    try:
        df = pd.read_sql_query(text('SELECT symbol FROM universe WHERE included = TRUE ORDER BY symbol'), engine)
        return df['symbol'].tolist()
    except Exception:
        return []


def _load_prices_batch(symbols: List[str], start_ts: pd.Timestamp) -> pd.DataFrame:
    """Load daily price data for a batch of symbols starting from a given timestamp."""
    if not symbols:
        return pd.DataFrame(columns=['symbol', 'ts', 'open', 'close', 'adj_close', 'volume'])
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
    """Load daily returns for a benchmark symbol (e.g., IWM)."""
    try:
        df = pd.read_sql_query(text(
            f"SELECT ts, {price_expr()} AS px FROM daily_bars WHERE symbol=:m ORDER BY ts"
        ), engine, params={'m': market_symbol}, parse_dates=['ts'])
        if df.empty:
            return pd.DataFrame(columns=['ts', 'mret'])
        df['mret'] = df['px'].pct_change()
        return df[['ts', 'mret']]
    except Exception:
        return pd.DataFrame(columns=['ts', 'mret'])


def _load_fundamentals(symbols: List[str]) -> pd.DataFrame:
    """Load point‑in‑time fundamental ratios for the given symbols.

    The query includes return_on_equity in addition to other standard ratios.
    """
    if not symbols:
        return pd.DataFrame(columns=['symbol', 'as_of'])
    sql = (
        'SELECT symbol, as_of, pe_ttm, pb, ps_ttm, debt_to_equity, '
        'return_on_assets, return_on_equity, gross_margins, profit_margins, current_ratio '
        'FROM fundamentals WHERE symbol IN :syms ORDER BY symbol, as_of'
    )
    stmt = text(sql).bindparams(bindparam('syms', expanding=True))
    return pd.read_sql_query(stmt, engine, params={'syms': tuple(symbols)}, parse_dates=['as_of'])


def _load_shares_outstanding(symbols: List[str]) -> pd.DataFrame:
    """Load shares outstanding (PIT) for the given symbols."""
    if not symbols:
        return pd.DataFrame(columns=['symbol', 'as_of', 'shares'])
    sql = 'SELECT symbol, as_of, shares FROM shares_outstanding WHERE symbol IN :syms ORDER BY symbol, as_of'
    stmt = text(sql).bindparams(bindparam('syms', expanding=True))
    return pd.read_sql_query(stmt, engine, params={'syms': tuple(symbols)}, parse_dates=['as_of'])


def build_features(batch_size: int = 200, warmup_days: int = 90) -> pd.DataFrame:
    """Incrementally compute and upsert features for the universe.

    This function processes the universe in batches, computing technical,
    microstructure, and fundamental features.  New rows are only computed for
    timestamps strictly after the last stored feature per symbol (except for
    a warmup period).

    Parameters
    ----------
    batch_size: int
        Number of symbols to process per batch.  Lower values reduce memory
        usage but increase runtime.
    warmup_days: int
        Number of days of historical data to include prior to the last feature
        timestamp for rolling calculations.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of new feature rows.
    """
    log.info("Starting feature build process (Incremental, PIT).")
    syms = _symbols()
    if not syms:
        log.warning("Universe is empty. Cannot build features.")
        return pd.DataFrame()
    last_map = _last_feature_dates()
    new_rows: list[pd.DataFrame] = []
    mkt = _load_market_returns("IWM")
    for i in range(0, len(syms), batch_size):
        bsyms = syms[i:i + batch_size]
        log.info(f"Processing batch {i // batch_size + 1} / {int(np.ceil(len(syms) / batch_size))} (Symbols: {len(bsyms)})")
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
        # Choose adjusted close when available
        px['price_feat'] = px['adj_close'].where(px['adj_close'].notna(), px['close'])
        out_frames: list[pd.DataFrame] = []
        # Merge market returns for beta/idio‑vol computations
        if not mkt.empty:
            px = px.merge(mkt, on='ts', how='left')
        for sym, g in px.groupby('symbol'):
            g = g.sort_values('ts').copy()
            p = g['price_feat']
            # Price, momentum and volatility
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
            # PIT shares to compute market cap
            shs_sym = shs[shs['symbol'] == sym][['as_of', 'shares']].sort_values('as_of')
            if not shs_sym.empty:
                g = pd.merge_asof(
                    g.sort_values('ts'),
                    shs_sym.rename(columns={'as_of': 'ts_shs'}).sort_values('ts_shs'),
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
            # Merge PIT fundamentals as of the latest available date
            f_sym = fnd[fnd['symbol'] == sym].drop(columns=['symbol']).sort_values('as_of')
            if not f_sym.empty:
                g = pd.merge_asof(
                    g.sort_values('ts'),
                    f_sym.rename(columns={'as_of': 'ts_fnd'}).sort_values('ts_fnd'),
                    left_on='ts', right_on='ts_fnd', direction='backward'
                )
            # Ensure all fundamental columns exist (fill with NaN if missing)
            for col in ['pe_ttm', 'pb', 'ps_ttm', 'debt_to_equity', 'return_on_assets', 'return_on_equity', 'gross_margins', 'profit_margins', 'current_ratio']:
                if col not in g.columns:
                    g[col] = np.nan
            # Rolling beta vs. market and macro features
            if 'mret' in g.columns and not g['mret'].isna().all():
                # Beta computed over a 63‑day window
                cov = g['ret_1d'].rolling(63).cov(g['mret'])
                var = g['mret'].rolling(63).var()
                g['beta_63'] = cov / var.replace(0, np.nan)
                # Macro features derived from the benchmark returns
                g['mkt_ret_1d'] = g['mret']
                g['mkt_ret_5d'] = g['mret'].rolling(5).sum()
                g['mkt_vol_21'] = g['mret'].rolling(21).std()
            else:
                # If market returns are missing, fall back to beta=1 and NaNs for macro
                g['beta_63'] = 1.0
                g['mkt_ret_1d'] = np.nan
                g['mkt_ret_5d'] = np.nan
                g['mkt_vol_21'] = np.nan
                if 'mret' not in g.columns:
                    log.debug(f"No market returns data, using beta=1.0 for {sym}")
            # Discard rows prior to last computed feature for this symbol
            last_ts = last_map.get(sym)
            if last_ts is not None and not pd.isna(last_ts):
                g = g[g['ts'] > pd.Timestamp(last_ts)]
            # Assemble feature columns
            fcols = [
                'symbol', 'ts',
                'ret_1d', 'ret_5d', 'ret_21d', 'mom_21', 'mom_63', 'vol_21',
                'rsi_14', 'turnover_21', 'size_ln', 'adv_usd_21',
                'overnight_gap', 'illiq_21', 'beta_63',
                'f_pe_ttm', 'f_pb', 'f_ps_ttm', 'f_debt_to_equity',
                'f_roa', 'f_roe', 'f_gm', 'f_profit_margin', 'f_current_ratio',
                # Market‑level macro features
                'mkt_ret_1d', 'mkt_ret_5d', 'mkt_vol_21'
            ]
            g['f_pe_ttm'] = g.get('pe_ttm')
            g['f_pb'] = g.get('pb')
            g['f_ps_ttm'] = g.get('ps_ttm')
            g['f_debt_to_equity'] = g.get('debt_to_equity')
            g['f_roa'] = g.get('return_on_assets')
            g['f_roe'] = g.get('return_on_equity')
            g['f_gm'] = g.get('gross_margins')
            g['f_profit_margin'] = g.get('profit_margins')
            g['f_current_ratio'] = g.get('current_ratio')
            # Drop rows with missing core returns
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

            # ------------------------------------------------------------------
            # Cross‑sectional z‑scores
            #
            # To augment the feature set with cross‑sectional information, compute
            # a z‑score for selected features within each timestamp.  These
            # standardised values capture a stock’s position relative to the
            # universe on that day (e.g. a high momentum rank vs. peers).  The
            # resulting columns are prefixed with ``cs_z_``.  This is optional
            # and computed here so downstream models can leverage relative
            # rankings.  We only compute z‑scores when there are at least 10
            # securities available on a given date to avoid unstable statistics.
            # ------------------------------------------------------------------
            features_for_cs = [
                'mom_21', 'mom_63', 'vol_21', 'rsi_14', 'turnover_21',
                'size_ln', 'adv_usd_21', 'beta_63',
                'f_pe_ttm', 'f_pb', 'f_ps_ttm', 'f_debt_to_equity',
                'f_roa', 'f_roe', 'f_gm', 'f_profit_margin', 'f_current_ratio'
            ]
            # Ensure all columns exist before attempting to compute z‑scores
            missing_cols = [c for c in features_for_cs if c not in feats.columns]
            for c in missing_cols:
                feats[c] = np.nan
            def _zscore(x: pd.Series) -> pd.Series:
                m = x.mean()
                s = x.std(ddof=0)
                return (x - m) / (s + 1e-8)
            # Group by timestamp and apply z‑score; only compute when group size
            # >= 10 to avoid small‑sample noise.  Otherwise fill with NaN.
            for col in features_for_cs:
                zcol = f'cs_z_{col}'
                feats[zcol] = feats.groupby('ts')[col].transform(
                    lambda s: _zscore(s) if len(s) >= 10 else pd.Series(np.nan, index=s.index)
                )

            upsert_dataframe(feats, Feature, ['symbol', 'ts'], chunk_size=200)
            new_rows.append(feats)
            log.info(f"Batch completed. New rows: {len(feats)}")
    return pd.concat(new_rows, ignore_index=True) if new_rows else pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_features()