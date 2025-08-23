from __future__ import annotations
import numpy as np
import pandas as pd
from os import cpu_count
from typing import Dict
from sqlalchemy import text
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from db import engine, upsert_dataframe, Prediction, BacktestEquity
from config import BACKTEST_START, TARGET_HORIZON_DAYS, TOP_N, SLIPPAGE_BPS, BLEND_WEIGHTS, GROSS_LEVERAGE, NET_EXPOSURE, MAX_POSITION_WEIGHT, DAILY_REBALANCE_COST_BPS

PRICE_SQL = "SELECT symbol, ts, COALESCE(adj_close, close) AS px FROM daily_bars WHERE ts >= :start AND ts <= :end"
FEATURE_COLS = ["ret_1d","ret_5d","ret_21d","mom_21","mom_63","vol_21","rsi_14","turnover_21","size_ln",
                "f_pe_ttm","f_pb","f_ps_ttm","f_debt_to_equity","f_roa","f_gm","f_profit_margin","f_current_ratio"]
ESSENTIAL_COLS = ["ret_1d","ret_5d","ret_21d","mom_21","mom_63","vol_21","rsi_14"]

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
        "xgb": Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("xgb", XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                              subsample=0.8, colsample_bytree=0.8,
                                              random_state=42, n_jobs=_xgb_threads(), tree_method="hist"))]),
        "rf": Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("rf", RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=_xgb_threads()))]),
        "ridge": Pipeline([("imputer", SimpleImputer(strategy="median")),
                           ("scaler", StandardScaler()),
                           ("ridge", Ridge(alpha=1.0))])
    }

def _parse_blend_weights(s: str):
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

def train_and_predict_all_models(window_years: int = 4):
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

    train_df = df.dropna(subset=ESSENTIAL_COLS + ["fwd_ret"]).copy()
    latest_df = feats[feats["ts"] == latest_ts].copy()
    if train_df.empty or latest_df.empty:
        return {}

    X = train_df[FEATURE_COLS].values
    y = train_df["fwd_ret"].values
    X_latest = latest_df[FEATURE_COLS].values

    models = _model_specs()
    outputs = {}
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
        upsert_dataframe(out[["symbol","ts","y_pred","model_version"]], Prediction, ["symbol","ts","model_version"])

    from config import BLEND_WEIGHTS
    w = _parse_blend_weights(BLEND_WEIGHTS)
    if w:
        sym = latest_df["symbol"].tolist()
        blend = np.zeros(len(sym), dtype=float)
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

def run_walkforward_backtest(rebalance_every: int = 5, window_years: int = 6, allow_shorts: bool = False):
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
    active_weights = {}
    scheduled_exits = {}
    prev_scaled_w = None  # for daily turnover costs

    def tranche_weights(pred_df: pd.DataFrame):
        pred_df = pred_df.sort_values("y_pred", ascending=False)
        longs = pred_df.head(TOP_N)["symbol"].tolist()
        nL = len(longs)
        if nL > 0:
            per_w = min(1.0 / nL, MAX_POSITION_WEIGHT)
            return {s: per_w for s in longs}
        return {}

    def model_for_window():
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("xgb", XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=max(1,(cpu_count() or 2)-1), tree_method="hist"))])

    for i, d in enumerate(dates[:-H]):
        d = pd.Timestamp(d)
        if i % rebalance_every == 0:
            train_start = max(pd.Timestamp(BACKTEST_START), d - pd.DateOffset(years=window_years))
            join_px = px_all[["symbol","ts","px"]].copy()
            px_fwd = join_px.copy()
            px_fwd["px_fwd"] = px_fwd.groupby("symbol")["px"].shift(-H)
            px_fwd["fwd_ret"] = (px_fwd["px_fwd"] / px_fwd["px"]) - 1.0
            joined = feats_all.merge(px_fwd[["symbol","ts","fwd_ret"]], on=["symbol","ts"], how="left")
            train_df = joined[(joined["ts"] < d) & (joined["ts"] >= train_start) & (~joined["fwd_ret"].isna())]
            train_df = train_df.dropna(subset=ESSENTIAL_COLS)
            today_feats = feats_all[feats_all["ts"] == d]
            if not train_df.empty and not today_feats.empty:
                X = train_df[FEATURE_COLS].values
                y = train_df["fwd_ret"].values
                pipe = model_for_window()
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
        if day_px.shape[1] == 2 and active_weights:
            ret = (day_px.iloc[:,1] / day_px.iloc[:,0]) - 1.0
            ret = ret.fillna(0.0)
            w = pd.Series(active_weights, dtype=float).reindex(ret.index).fillna(0.0)

            # Scale to target gross & net exposure each day
            tgtL = max(0.0, (GROSS_LEVERAGE + NET_EXPOSURE) / 2.0)
            tgtS = max(0.0, (GROSS_LEVERAGE - NET_EXPOSURE) / 2.0)
            curL = float(w[w > 0].sum()); curS = float(-w[w < 0].sum())
            w_scaled = w.copy()
            if curL > 0: w_scaled[w_scaled > 0] *= (tgtL / curL)
            if curS > 0: w_scaled[w_scaled < 0] *= (tgtS / curS)

            # Charge daily rebalance costs based on turnover vs prior scaled weights
            if prev_scaled_w is not None:
                joint = w_scaled.reindex(prev_scaled_w.index).fillna(0.0)
                prev = prev_scaled_w.reindex(w_scaled.index).fillna(0.0)
                turnover = float((joint - prev).abs().sum())
                cost = (DAILY_REBALANCE_COST_BPS / 10000.0) * turnover
                equity *= (1.0 - cost)

            port_ret = float((w_scaled * ret).sum())
            equity *= (1.0 + port_ret)
            prev_scaled_w = w_scaled

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
