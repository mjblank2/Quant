from __future__ import annotations
import numpy as np, pandas as pd, copy, logging, os
from os import cpu_count
from typing import Dict, Any
from sqlalchemy import text
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from db import engine, upsert_dataframe, Prediction
from models.transformers import CrossSectionalNormalizer
from config import (BACKTEST_START, TARGET_HORIZON_DAYS, BLEND_WEIGHTS,
                    SECTOR_NEUTRALIZE,)
from risk.sector import build_sector_dummies
from risk.risk_model import neutralize_with_sectors
from models.regime import classify_regime, gate_blend_weights
from data.universe_history import gate_training_with_universe
log = logging.getLogger(__name__)

FEATURE_COLS = [
    "ret_1d","ret_5d","ret_21d","mom_21","mom_63","vol_21","rsi_14",
    "turnover_21","size_ln","overnight_gap","illiq_21","beta_63",
    "f_pe_ttm","f_pb","f_ps_ttm","f_debt_to_equity","f_roa","f_gm","f_profit_margin","f_current_ratio",
    # sparse high-impact event features:
    "pead_event","pead_surprise_eps","pead_surprise_rev","russell_inout"
]
TCA_COLS = ["vol_21", "adv_usd_21"]

def _latest_feature_date():
    try:
        df = pd.read_sql_query(text("SELECT MAX(ts) AS max_ts FROM features"), engine, parse_dates=["max_ts"])
        return df.iloc[0,0]
    except Exception:
        return None

def _load_features_window(start_ts, end_ts):
    cols = list(set(FEATURE_COLS + TCA_COLS + ["symbol","ts"]))
    cols_str = ", ".join(cols)
    sql = f"SELECT {cols_str} FROM features WHERE ts >= :start AND ts <= :end"
    try:
        return pd.read_sql_query(text(sql), engine, params={"start": start_ts.date(), "end": end_ts.date()}, parse_dates=["ts"]).sort_values(["ts","symbol"])
    except Exception as e:
        log.error(f"Failed to load features window: {e}")
        return pd.DataFrame(columns=cols)

def _load_prices_window(start_ts, end_ts):
    try:
        return pd.read_sql_query(text("SELECT symbol, ts, COALESCE(adj_close, close) AS px FROM daily_bars WHERE ts>=:s AND ts<=:e"),
                                 engine, params={'s': start_ts.date(), 'e': end_ts.date()}, parse_dates=['ts']).sort_values(['symbol','ts'])
    except Exception:
        return pd.DataFrame(columns=['symbol','ts','px'])

def _load_altsignals(start_ts, end_ts):
    try:
        df = pd.read_sql_query(text("SELECT symbol, ts, name, value FROM alt_signals WHERE ts>=:s AND ts<=:e"),
                               engine, params={'s': start_ts.date(), 'e': end_ts.date()}, parse_dates=['ts'])
    except Exception:
        return pd.DataFrame(columns=['symbol','ts','name','value'])
    if df.empty: return df
    # pivot sparse signals
    piv = df.pivot_table(index=['symbol','ts'], columns='name', values='value', aggfunc='last').reset_index()
    return piv

def _xgb_threads() -> int:
    c = cpu_count() or 2
    return max(1, c-1)

def _define_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("normalizer", CrossSectionalNormalizer(winsorize_tails=0.05)),
                     ("model", estimator)])

def _model_specs() -> Dict[str, Pipeline]:
    return {
        "xgb": _define_pipeline(XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                                             subsample=0.8, colsample_bytree=0.8,
                                             random_state=42, n_jobs=_xgb_threads(), tree_method="hist")),
        "rf": _define_pipeline(RandomForestRegressor(n_estimators=350, max_depth=10, random_state=42, n_jobs=_xgb_threads())),
        "ridge": _define_pipeline(Ridge(alpha=1.0)),
    }

def _parse_blend_weights(s: str) -> Dict[str, float]:
    out = {}
    for part in s.split(","):
        if ":" in part:
            k,v = part.split(":",1)
            try: out[k.strip()] = float(v)
            except: pass
    t = sum(out.values()) or 1.0
    return {k: v/t for k,v in out.items()}

def _ic_by_model(train_df: pd.DataFrame, feature_cols: list[str]) -> Dict[str,float]:
    models = _model_specs()
    if train_df.empty: return {}
    ics = {}
    # use last ~6 months
    recent_mask = train_df['ts'] >= (train_df['ts'].max() - pd.Timedelta(days=180))
    Xr = train_df.loc[recent_mask, feature_cols]; yr = train_df.loc[recent_mask, 'fwd_ret']
    if Xr.empty:
        Xr, yr = train_df[feature_cols], train_df['fwd_ret']
    for name, pipe in models.items():
        try:
            p2 = copy.deepcopy(pipe)
            p2.fit(train_df[feature_cols], train_df['fwd_ret'].values)
            preds = pd.Series(p2.predict(Xr), index=Xr.index)
            tmp = train_df.loc[recent_mask, ['ts']].copy()
            tmp['pred'] = preds.values; tmp['y'] = yr.values
            ic_by_date = tmp.groupby('ts').apply(lambda d: d['pred'].corr(d['y'], method='spearman'))
            ics[name] = float(ic_by_date.mean())
        except Exception:
            ics[name] = 0.0
    pos = {k: max(0.0, v) for k,v in ics.items()}
    s = sum(pos.values()) or 1.0
    return {k: v/s for k,v in pos.items()}

def train_and_predict_all_models(window_years: int = 4):
    log.info("Starting live training and prediction (v17).")
    latest_ts = _latest_feature_date()
    if latest_ts is None or pd.isna(latest_ts):
        log.warning("No features available. Cannot train models.")
        return {}

    start_ts = max(pd.Timestamp(BACKTEST_START), latest_ts - pd.DateOffset(years=window_years))
    feats = _load_features_window(start_ts, latest_ts)
    if feats.empty: return {}

    # Merge sparse AltSignals
    alts = _load_altsignals(start_ts, latest_ts)
    if not alts.empty:
        feats = feats.merge(alts, on=['symbol','ts'], how='left')

    # Survivorship gating using snapshots (if any)
    try:
        feats = gate_training_with_universe(feats)
    except Exception as e:
        log.info(f"Universe gating skipped: {e}")

    px = _load_prices_window(start_ts, latest_ts + pd.Timedelta(days=TARGET_HORIZON_DAYS+5))
    if px.empty: return {}

    px = px.sort_values(['symbol','ts']).copy()
    px['px_fwd'] = px.groupby('symbol')['px'].shift(-TARGET_HORIZON_DAYS)
    px['fwd_ret'] = (px['px_fwd'] / px['px']) - 1.0
    df = feats.merge(px[['symbol','ts','fwd_ret']], on=['symbol','ts'], how='left')
    train_df = df.dropna(subset=['fwd_ret']).copy()
    latest_df = feats[feats['ts'] == latest_ts].copy()
    if train_df.empty or latest_df.empty:
        log.warning("Training or prediction sets are empty."); return {}

    # Build matrices (fill missing sparse features with 0)
    for c in FEATURE_COLS:
        if c not in train_df.columns: train_df[c] = 0.0
        if c not in latest_df.columns: latest_df[c] = 0.0

    X = train_df[FEATURE_COLS]; y = train_df['fwd_ret'].values
    X_latest = latest_df[FEATURE_COLS]
    models = _model_specs()
    outputs, preds_dict = {}, {}

    # Blend weights: config + IC adaptive
    from config import BLEND_WEIGHTS, REGIME_GATING
    blend_w = _parse_blend_weights(BLEND_WEIGHTS)
    ic_w = _ic_by_model(train_df[['ts','symbol','fwd_ret'] + FEATURE_COLS], FEATURE_COLS)
    if ic_w:
        keys = set(blend_w) | set(ic_w)
        combined = {k: 0.5*blend_w.get(k,0)+0.5*ic_w.get(k,0) for k in keys}
        s = sum(combined.values()) or 1.0
        blend_w = {k: v/s for k,v in combined.items()}

    # Regime gating
    if REGIME_GATING:
        regime = classify_regime(latest_ts)
        blend_w = gate_blend_weights(blend_w, regime)
        log.info(f"Regime: {regime}; Blend weights gated: {blend_w}")

    # Fit & predict base models
    for name, pipe in models.items():
        p2 = copy.deepcopy(pipe)
        p2.fit(X, y)
        p = p2.predict(X_latest)
        out = latest_df[['symbol']].copy()
        out['ts'] = latest_ts; out['y_pred'] = p.astype(float); out['model_version'] = f"{name}_v1"
        preds_dict[name] = out.set_index('symbol')['y_pred']
        outputs[name] = out.copy()
        upsert_dataframe(out[['symbol','ts','y_pred','model_version']], Prediction, ['symbol','ts','model_version'])

    # Blended + neutralized
    if blend_w:
        sym_index = latest_df['symbol']
        blend = np.zeros(len(sym_index))
        for name, w in blend_w.items():
            if name in preds_dict:
                series = preds_dict[name].reindex(sym_index).fillna(0.0).values
                blend += w * series
        out = latest_df[['symbol']].copy()
        out['ts'] = latest_ts; out['y_pred'] = blend.astype(float); out['model_version'] = 'blend_raw_v17'
        outputs['blend_raw'] = out.copy()
        upsert_dataframe(out[['symbol','ts','y_pred','model_version']], Prediction, ['symbol','ts','model_version'])
        try:
            fac = latest_df.set_index('symbol')[['size_ln','mom_21','turnover_21','beta_63'] if 'beta_63' in latest_df.columns else ['size_ln','mom_21','turnover_21']]
            pred_series = pd.Series(blend, index=sym_index)
            from risk.sector import build_sector_dummies
            sd = build_sector_dummies(sym_index.tolist(), latest_ts) if True else None
            resid = neutralize_with_sectors(pred_series, fac, sd)
            out2 = latest_df[['symbol']].copy()
            out2['ts'] = latest_ts; out2['y_pred'] = resid.values; out2['model_version'] = 'blend_v1'
            outputs['blend'] = out2.copy()
            upsert_dataframe(out2[['symbol','ts','y_pred','model_version']], Prediction, ['symbol','ts','model_version'])
        except Exception as e:
            log.warning(f"Neutralization failed: {e}; fallback to raw blend.")
            out_fallback = out.copy(); out_fallback['model_version']='blend_v1'
            upsert_dataframe(out_fallback[['symbol','ts','y_pred','model_version']], Prediction, ['symbol','ts','model_version'])
    log.info("Live training and prediction complete (v17).")
    return outputs
