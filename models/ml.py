from __future__ import annotations
import numpy as np
import pandas as pd
from os import cpu_count
from typing import Dict, Any
from sqlalchemy import text
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import copy
from db import engine, upsert_dataframe, Prediction, BacktestEquity
from models.transformers import CrossSectionalNormalizer
from config import (
    BACKTEST_START, TARGET_HORIZON_DAYS, TOP_N, MARKET_IMPACT_BETA, BLEND_WEIGHTS,
    GROSS_LEVERAGE, NET_EXPOSURE, MAX_POSITION_WEIGHT, ALLOW_SHORTS,
    SPREAD_BPS, COMMISSION_BPS, SECTOR_NEUTRALIZE
)
import logging
from data.universe_history import gate_training_with_universe
from models.regime import realized_vol, liquidity_breadth, classify_regime, gate_blend_weights
from risk.sector import build_sector_dummies
from risk.risk_model import neutralize_with_sectors
from short.borrow import borrow_state_asof
import os

log = logging.getLogger(__name__)

PRICE_SQL = "SELECT symbol, ts, COALESCE(adj_close, close) AS px FROM daily_bars WHERE ts >= :start AND ts <= :end"

# Extended feature set for v16
FEATURE_COLS = [
    'overnight_gap','illiq_21','beta_63',  # ensure v16 microstructure
    
    "ret_1d","ret_5d","ret_21d","mom_21","mom_63","vol_21","rsi_14",
    "turnover_21","size_ln","overnight_gap","illiq_21","beta_63",
    "f_pe_ttm","f_pb","f_ps_ttm","f_debt_to_equity","f_roa","f_gm","f_profit_margin","f_current_ratio"
]
TCA_COLS = ["vol_21", "adv_usd_21"]

def _latest_feature_date() -> pd.Timestamp | None:
    try:
        df = pd.read_sql_query(text("SELECT MAX(ts) AS max_ts FROM features"), engine, parse_dates=["max_ts"])
        return df.iloc[0,0]
    except Exception:
        return None

def _load_features_window(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    cols = list(set(FEATURE_COLS + TCA_COLS + ["symbol","ts"]))
    cols_str = ", ".join(cols)
    sql = f"SELECT {cols_str} FROM features WHERE ts >= :start AND ts <= :end"
    try:
        return pd.read_sql_query(text(sql), engine, params={"start": start_ts.date(), "end": end_ts.date()}, parse_dates=["ts"]).sort_values(["ts","symbol"])
    except Exception as e:
        log.error(f"Failed to load features window: {e}")
        return pd.DataFrame(columns=cols)

def _load_prices_window


def _load_alt_signals_window(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    try:
        cols = ['symbol','ts','name','value']
        df = pd.read_sql_query(text(
            "SELECT symbol, ts, name, value FROM alt_signals WHERE ts >= :start AND ts <= :end"
        ), engine, params={'start': start_ts.date(), 'end': end_ts.date()}, parse_dates=['ts'])
        if df.empty:
            return pd.DataFrame(columns=['symbol','ts'])
        piv = df.pivot_table(index=['symbol','ts'], columns='name', values='value', aggfunc='last')
        piv = piv.reset_index()
        return piv
    except Exception:
        return pd.DataFrame(columns=['symbol','ts'])
(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    try:
        return pd.read_sql_query(text(PRICE_SQL), engine, params={"start": start_ts.date(), "end": end_ts.date()}, parse_dates=["ts"]).sort_values(["symbol","ts"])
    except Exception as e:
        log.error(f"Failed to load prices window: {e}")
        return pd.DataFrame(columns=["symbol", "ts", "px"])

def _xgb_threads() -> int:
    c = cpu_count() or 2
    return max(1, c - 1)

def _define_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([
        ("normalizer", CrossSectionalNormalizer(winsorize_tails=0.05)),
        ("model", estimator)
    ])

def _model_specs() -> Dict[str, Pipeline]:
    return {
        "xgb": _define_pipeline(
            XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=42, n_jobs=_xgb_threads(), tree_method="hist")
        ),
        "rf": _define_pipeline(
            RandomForestRegressor(n_estimators=350, max_depth=10, random_state=42, n_jobs=_xgb_threads())
        ),
        "ridge": _define_pipeline(Ridge(alpha=1.0))
    }

def _parse_blend_weights(s: str) -> Dict[str, float]:
    out = {}
    for part in s.split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                out[k.strip()] = float(v)
            except Exception:
                pass
    total = sum(out.values())
    if total == 0:
        return {}
    for k in list(out.keys()):
        out[k] /= total
    return out

def calculate_market_impact(trade_notional: float, sigma: float, adv: float, beta: float = MARKET_IMPACT_BETA) -> float:
    if adv <= 0 or not np.isfinite(adv) or not np.isfinite(sigma) or sigma <= 0:
        return 0.0
    participation_rate = min(abs(trade_notional) / adv, 0.30)
    impact_rate = beta * sigma * np.sqrt(participation_rate)  # ~bps in decimal
    return abs(trade_notional) * impact_rate

def _ic_by_model(train_df: pd.DataFrame, feature_cols: list[str]) -> Dict[str,float]:
    """Train quick models and compute trailing IC for each model to adapt blend weights."""
    models = _model_specs()
    # Use last ~6 months for IC calc
    if train_df.empty:
        return {}
    # group by date (cross-sectional IC)
    ics: Dict[str, float] = {}
    X = train_df[feature_cols]; y = train_df["fwd_ret"].values
    # Choose last ~120 trading days for stability
    recent_mask = train_df["ts"] >= (train_df["ts"].max() - pd.Timedelta(days=180))
    Xr = train_df.loc[recent_mask, feature_cols]; yr = train_df.loc[recent_mask, "fwd_ret"]
    if Xr.empty:
        Xr, yr = X, y
    for name, pipe in models.items():
        try:
            pipe2 = copy.deepcopy(pipe)
            pipe2.fit(X, y)
            preds = pd.Series(pipe2.predict(Xr), index=train_df.loc[recent_mask].index)
            tmp = train_df.loc[recent_mask, ["ts"]].copy(); tmp["pred"]=preds.values; tmp["y"]=yr.values
            # Spearman IC per date then average
            ic_by_date = tmp.groupby("ts").apply(lambda d: d["pred"].corr(d["y"], method="spearman"))
            ics[name] = float(ic_by_date.mean())
        except Exception:
            ics[name] = 0.0
    # Normalize to 0..1 weights (clip negatives)
    min0 = {k: max(0.0, v) for k,v in ics.items()}
    s = sum(min0.values()) or 1.0
    return {k: v/s for k,v in min0.items()}

def train_and_predict_all_models(window_years: int = 4):
    log.info("Starting live training and prediction (v16).")
    latest_ts = _latest_feature_date()
    if latest_ts is None or pd.isna(latest_ts):
        log.warning("No features available. Cannot train models.")
        return {}

    start_ts = max(pd.Timestamp(BACKTEST_START), latest_ts - pd.DateOffset(years=window_years))
    feats = _load_features_window(start_ts, latest_ts)
    if feats.empty:
        return {}
    px = _load_prices_window(start_ts, latest_ts + pd.Timedelta(days=TARGET_HORIZON_DAYS+5))
    if px.empty:
        return {}

    px = px.sort_values(["symbol","ts"]).copy()
    px["px_fwd"] = px.groupby("symbol")["px"].shift(-TARGET_HORIZON_DAYS)
    px["fwd_ret"] = (px["px_fwd"] / px["px"]) - 1.0
    df = feats.merge(px[["symbol","ts","fwd_ret"]], on=["symbol","ts"], how="left")
    train_df = df.dropna(subset=["fwd_ret"]).copy()
    try:
        train_df = gate_training_with_universe(train_df)
    except Exception as _:
        pass
    latest_df = feats[feats["ts"] == latest_ts].copy()
    if train_df.empty or latest_df.empty:
        log.warning("Training or prediction dataframes are empty.")
        return {}

    X = train_df[FEATURE_COLS]; y = train_df["fwd_ret"].values
    X_latest = latest_df[FEATURE_COLS]

    models = _model_specs()
    outputs = {}
    preds_dict = {}

    # IC-adaptive blend weights (fallback to BLEND_WEIGHTS if unavailable)
    ic_w = _ic_by_model(train_df[["ts","symbol","fwd_ret"] + FEATURE_COLS].copy(), FEATURE_COLS)
    blend_w = _parse_blend_weights(BLEND_WEIGHTS)
    if ic_w:
        # combine: 50% config, 50% IC-adaptive
        # Regime gating (vol/liquidity)
        try:
            rv = realized_vol('IWM', 21)
            liq = liquidity_breadth(21)
            regime = classify_regime(rv, liq)
            blend_w = gate_blend_weights(blend_w, regime)
            log.info(f"Regime={regime} vol={rv:.4f if rv else float('nan')} liq={liq:.0f if liq else float('nan')} -> gated weights {blend_w}")
        except Exception as _:
            pass
        keys = set(blend_w) | set(ic_w)
        combined = {k: 0.5*blend_w.get(k, 0) + 0.5*ic_w.get(k, 0) for k in keys}
        s = sum(combined.values()) or 1.0
        blend_w = {k: v/s for k,v in combined.items()}
    log.info(f"Blend weights (effective): {blend_w}")

    for name, pipe in models.items():
        pipe2 = copy.deepcopy(pipe)
        pipe2.fit(X, y)
        p = pipe2.predict(X_latest)

        out = latest_df[["symbol"]].copy()
        out["ts"] = latest_ts
        out["y_pred"] = p.astype(float)
        out["model_version"] = f"{name}_v1"
        preds_dict[name] = out.set_index("symbol")["y_pred"]
        outputs[name] = out.copy()
        upsert_dataframe(out[["symbol","ts","y_pred","model_version"]], Prediction, ["symbol","ts","model_version"])

    # Blended + neutralized (sectors + factors)
    if blend_w:
        sym_index = latest_df["symbol"]
        blend = np.zeros(len(sym_index))
        for name, weight in blend_w.items():
            if name in preds_dict:
                series = preds_dict[name].reindex(sym_index).fillna(0.0).values
                blend += weight * series
        out = latest_df[["symbol"]].copy()
        out["ts"] = latest_ts
        out["y_pred"] = blend.astype(float)
        out["model_version"] = "blend_raw_v16"
        outputs["blend_raw"] = out.copy()
        upsert_dataframe(out[["symbol","ts","y_pred","model_version"]], Prediction, ["symbol","ts","model_version"])

        try:
            fac = latest_df.set_index("symbol")[["size_ln","mom_21","turnover_21","beta_63"]]
            pred_series = pd.Series(blend, index=sym_index)
            secD = build_sector_dummies(sym_index.tolist(), latest_ts) if SECTOR_NEUTRALIZE else None
            resid = neutralize_with_sectors(pred_series, fac, secD)
            out2 = latest_df[["symbol"]].copy(); out2["ts"]=latest_ts; out2["y_pred"]=resid.values; out2["model_version"]="blend_v1"
            outputs["blend"] = out2.copy()
            upsert_dataframe(out2[["symbol","ts","y_pred","model_version"]], Prediction, ["symbol","ts","model_version"])
        except Exception as e:
            log.warning(f"Neutralization failed: {e}; falling back to raw blend as blend_v1")
            out_fallback = out.copy(); out_fallback["model_version"]="blend_v1"
            upsert_dataframe(out_fallback[["symbol","ts","y_pred","model_version"]], Prediction, ["symbol","ts","model_version"])

    log.info("Live training and prediction complete (v16).")
    return outputs

# Backtest left as in v15 but can be extended to add spread/commission on top of impact if desired.
if __name__ == "__main__":
    logging.basic(level=logging.INFO)
    train_and_predict_all_models()
