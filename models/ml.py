from __future__ import annotations
import numpy as np
import pandas as pd
from os import cpu_count
from typing import Dict
from sqlalchemy import text
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from db import engine, upsert_dataframe, Prediction, BacktestEquity
from config import BACKTEST_START, TARGET_HORIZON_DAYS, TOP_N, SLIPPAGE_BPS, BLEND_WEIGHTS

PRICE_SQL = "SELECT symbol, ts, COALESCE(adj_close, close) AS px FROM daily_bars WHERE ts >= :start AND ts <= :end"
FEATURE_COLS = ["ret_1d","ret_5d","ret_21d","mom_21","mom_63","vol_21","rsi_14","turnover_21","size_ln",
                "f_pe_ttm","f_pb","f_ps_ttm","f_debt_to_equity","f_roa","f_gm","f_profit_margin","f_current_ratio"]

def _latest_feature_date() -> pd.Timestamp | None:
    df = pd.read_sql_query(text("SELECT MAX(ts) AS max_ts FROM features"), engine, parse_dates=["max_ts"])
    return df.iloc[0,0]

def _load_features_window(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    return pd.read_sql_query(
        text("SELECT * FROM features WHERE ts >= :start AND ts <= :end"),
        engine,
        params={"start": start_ts.date(), "end": end_ts.date()},
        parse_dates=["ts"]
    ).sort_values(["ts","symbol"])

def _load_prices_window(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    return pd.read_sql_query(text(PRICE_SQL), engine, params={"start": start_ts.date(), "end": end_ts.date()}, parse_dates=["ts"]).sort_values(["symbol","ts"])

def _xgb_threads() -> int:
    c = cpu_count() or 2
    return max(1, c - 1)

def _model_specs() -> Dict[str, Pipeline]:
    return {
        "xgb": Pipeline([("scaler", StandardScaler()),
                         ("xgb", XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                              subsample=0.8, colsample_bytree=0.8,
                                              random_state=42, n_jobs=_xgb_threads(), tree_method="hist"))]),
        "rf": Pipeline([("scaler", StandardScaler()),
                        ("rf", RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=_xgb_threads()))]),
        "ridge": Pipeline([("scaler", StandardScaler()),
                           ("ridge", Ridge(alpha=1.0, random_state=42))])
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
    total = sum(out.values()) or 1.0
    for k in list(out.keys()):
        out[k] /= total
    return out

def train_and_predict_all_models(window_years: int = 4) -> Dict[str, pd.DataFrame]:
    latest_ts = _latest_feature_date()
    if latest_ts is None or pd.isna(latest_ts):
        return {}

    start_ts = max(pd.Timestamp(BACKTEST_START), latest_ts - pd.DateOffset(years=window_years))
    feats = _load_features_window(start_ts, latest_ts)
    if feats.empty:
        return {}

    px = _load_prices_window(start_ts, latest_ts + pd.Timedelta(days=TARGET_HORIZON_DAYS+1))
    if px.empty:
        return {}

    px = px.sort_values(["symbol","ts"]).copy()
    px["px_fwd"] = px.groupby("symbol")["px"].shift(-TARGET_HORIZON_DAYS)
    px["fwd_ret"] = (px["px_fwd"] / px["px"]) - 1.0
    df = feats.merge(px[["symbol","ts","fwd_ret"]], on=["symbol","ts"], how="left")

    train_df = df.dropna(subset=FEATURE_COLS + ["fwd_ret"]).copy()
    latest_df = feats[feats["ts"] == latest_ts].copy()
    if train_df.empty or latest_df.empty:
        return {}

    X = train_df[FEATURE_COLS].values
    y = train_df["fwd_ret"].values
    X_latest = latest_df[FEATURE_COLS].values

    models = _model_specs()
    outputs: Dict[str, pd.DataFrame] = {}
    preds_dict = {}

    for name, pipe in models.items():
        pipe.fit(X, y)
        p = pipe.predict(X_latest)
        out = latest_df[["symbol"]].copy()
        out["ts"] = latest_ts
        out["y_pred"] = p.astype(float)
        out["model_version"] = f"{name}_v1"
        preds_dict[name] = out[["symbol","y_pred"]].set_index("symbol")["y_pred"]
        outputs[name] = out.copy()
        # Upsert by (symbol, ts) to keep only preferred/last model if schema PK is (symbol, ts)
        upsert_dataframe(out[["symbol","ts","y_pred","model_version"]], Prediction, ["symbol","ts","model_version"])

    # Blend
    w = _parse_blend_weights(BLEND_WEIGHTS)
    if w:
        sym = latest_df["symbol"].tolist()
        blend = np.zeros(len(sym))
        for name, weight in w.items():
            if name in preds_dict:
                series = preds_dict[name].reindex(sym).fillna(0.0).values
                blend += weight * series
        out = latest_df[["symbol"]].copy()
        out["ts"] = latest_ts
        out["y_pred"] = blend.astype(float)
        out["model_version"] = "blend_v1"
        outputs["blend"] = out.copy()
        upsert_dataframe(out[["symbol","ts","y_pred","model_version"]], Prediction, ["symbol","ts","model_version"])

    return outputs

def run_walkforward_backtest(rebalance_every: int = 5, window_years: int = 6, allow_shorts: bool = False) -> pd.DataFrame:
    feats_all = pd.read_sql_query(text("SELECT * FROM features WHERE ts >= :start"), engine, params={"start": BACKTEST_START}, parse_dates=["ts"]).sort_values(["ts","symbol"])
    if feats_all.empty:
        return pd.DataFrame(columns=["ts","equity","daily_return","drawdown"])
    px_all = pd.read_sql_query(text(PRICE_SQL), engine, params={"start": BACKTEST_START, "end": feats_all["ts"].max().date()}, parse_dates=["ts"]).sort_values(["symbol","ts"])
    if px_all.empty:
        return pd.DataFrame(columns=["ts","equity","daily_return","drawdown"])

    dates = sorted(feats_all["ts"].unique())
    if len(dates) < 100:
        return pd.DataFrame(columns=["ts","equity","daily_return","drawdown"])

    H = TARGET_HORIZON_DAYS
    equity = 1.0
    rows = []
    active_weights: dict[str, float] = {}
    scheduled_exits: dict[pd.Timestamp, dict[str, float]] = {}

    def tranche_weights(pred_df: pd.DataFrame) -> dict[str, float]:
        pred_df = pred_df.sort_values("y_pred", ascending=False)
        longs = pred_df.head(TOP_N)["symbol"].tolist()
        nL = len(longs) or 1
        per_w = min(1.0 / nL, 0.03)
        return {s: per_w for s in longs}

    from sklearn.pipeline import Pipeline  # already imported; local alias for clarity
    def model_for_window(train_df: pd.DataFrame) -> Pipeline:
        return _model_specs()["xgb"]

    for i, d in enumerate(dates[:-H]):
        d = pd.Timestamp(d)
        if i % rebalance_every == 0:
            train_start = max(pd.Timestamp(BACKTEST_START), d - pd.DateOffset(years=window_years))
            today_feats = feats_all[feats_all["ts"] == d]
            join_px = px_all[["symbol","ts","px"]]
            px_fwd = join_px.copy()
            px_fwd["px_fwd"] = px_fwd.groupby("symbol")["px"].shift(-H)
            px_fwd["fwd_ret"] = (px_fwd["px_fwd"] / px_fwd["px"]) - 1.0
            joined = feats_all.merge(px_fwd[["symbol","ts","fwd_ret"]], on=["symbol","ts"], how="left")
            train_df = joined[(joined["ts"] < d) & (joined["ts"] >= train_start) & (~joined["fwd_ret"].isna())]
            if not train_df.empty and not today_feats.empty:
                X = train_df[FEATURE_COLS].values
                y = train_df["fwd_ret"].values
                pipe = model_for_window(train_df)
                pipe.fit(X, y)
                preds = pipe.predict(today_feats[FEATURE_COLS].values)
                pred_df = today_feats[["symbol"]].copy()
                pred_df["y_pred"] = preds.astype(float)
                tw = tranche_weights(pred_df)
                slip = (SLIPPAGE_BPS / 10000.0) * sum(abs(v) for v in tw.values())
                equity *= (1.0 - slip)
                for s, wgt in tw.items():
                    active_weights[s] = active_weights.get(s, 0.0) + wgt
                exit_date = dates[min(i + H, len(dates)-1)]
                scheduled_exits.setdefault(pd.Timestamp(exit_date), {})
                scheduled_exits[pd.Timestamp(exit_date)].update(tw)

        if i == 0:
            rows.append({"ts": d.date(), "equity": equity})
            continue
        d_prev = dates[i-1]
        day_px = px_all[px_all["ts"].isin([pd.Timestamp(d_prev), d])].pivot(index="symbol", columns="ts", values="px")
        if day_px.shape[1] == 2:
            ret = (day_px.iloc[:,1] / day_px.iloc[:,0]) - 1.0
            weight_series = pd.Series(active_weights, dtype=float)
            weight_series = weight_series.reindex(ret.index).fillna(0.0)
            port_ret = float((weight_series * ret).sum())
            equity *= (1.0 + port_ret)
        rows.append({"ts": d.date(), "equity": equity})

        if pd.Timestamp(d) in scheduled_exits:
            tw = scheduled_exits[pd.Timestamp(d)]
            slip = (SLIPPAGE_BPS / 10000.0) * sum(abs(v) for v in tw.values())
            equity *= (1.0 - slip)
            for s, wgt in tw.items():
                active_weights[s] = active_weights.get(s, 0.0) - wgt
                if abs(active_weights.get(s, 0.0)) < 1e-12:
                    active_weights.pop(s, None)

    if not rows:
        return pd.DataFrame(columns=["ts","equity","daily_return","drawdown"])
    df = pd.DataFrame(rows).sort_values("ts")
    df["daily_return"] = df["equity"].pct_change().fillna(0.0)
    rolling_max = df["equity"].cummax()
    df["drawdown"] = df["equity"] / rolling_max - 1.0
    upsert_dataframe(df, BacktestEquity, ["ts"])
    return df
