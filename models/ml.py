from __future__ import annotations
import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from db import engine, upsert_dataframe, Prediction, BacktestEquity
from config import BACKTEST_START, TARGET_HORIZON_DAYS, TOP_N, SLIPPAGE_BPS

FEATURE_COLS = ["ret_1d","ret_5d","ret_21","mom_21","mom_63","vol_21","rsi_14","turnover_21","size_ln"]

def _latest_feature_date() -> pd.Timestamp | None:
    return pd.read_sql_query(text("SELECT MAX(ts) AS max_ts FROM features"), engine, parse_dates=["max_ts"]).iloc[0,0]

def _load_features_window(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    return pd.read_sql_query(
        text("SELECT * FROM features WHERE ts >= :start AND ts <= :end"),
        engine,
        params={"start": start_ts.date(), "end": end_ts.date()},
        parse_dates=["ts"]
    ).sort_values(["ts","symbol"])

def _load_prices_window(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    return pd.read_sql_query(
        text("SELECT symbol, ts, COALESCE(adj_close, close) AS px FROM daily_bars WHERE ts >= :start AND ts <= :end"),
        engine,
        params={"start": start_ts.date(), "end": end_ts.date()},
        parse_dates=["ts"]
    ).sort_values(["symbol","ts"])

def train_and_predict_latest(window_years: int = 4) -> pd.DataFrame:
    latest_ts = _latest_feature_date()
    if latest_ts is None or pd.isna(latest_ts):
        return pd.DataFrame(columns=["symbol","ts","y_pred"])

    start_ts = max(pd.Timestamp(BACKTEST_START), latest_ts - pd.DateOffset(years=window_years))

    feats = _load_features_window(start_ts, latest_ts)
    if feats.empty:
        return pd.DataFrame(columns=["symbol","ts","y_pred"])

    px = _load_prices_window(start_ts, latest_ts + pd.Timedelta(days=TARGET_HORIZON_DAYS+1))
    if px.empty:
        return pd.DataFrame(columns=["symbol","ts","y_pred"])

    # forward returns
    px = px.sort_values(["symbol","ts"]).copy()
    px["px_fwd"] = px.groupby("symbol")["px"].shift(-TARGET_HORIZON_DAYS)
    px["fwd_ret"] = (px["px_fwd"] / px["px"]) - 1.0
    df = feats.merge(px[["symbol","ts","fwd_ret"]], on=["symbol","ts"], how="left")

    # Train on rows with known future returns
    train_df = df.dropna(subset=FEATURE_COLS + ["fwd_ret"]).copy()
    if train_df.empty:
        return pd.DataFrame(columns=["symbol","ts","y_pred"])

    X = train_df[FEATURE_COLS].values
    y = train_df["fwd_ret"].values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=0, tree_method="hist"
        ))
    ])
    pipe.fit(X, y)

    latest_df = feats[feats["ts"] == latest_ts].copy()
    if latest_df.empty:
        return pd.DataFrame(columns=["symbol","ts","y_pred"])
    preds = pipe.predict(latest_df[FEATURE_COLS].values)
    out = latest_df[["symbol"]].copy()
    out["ts"] = latest_ts
    out["y_pred"] = preds.astype(float)
    upsert_dataframe(out, Prediction, ["symbol","ts"])
    return out

def run_backtest(window_years: int = 6) -> pd.DataFrame:
    # lightweight: rolling monthly retrain starting from BACKTEST_START (windowed to reduce memory)
    start_ts = pd.Timestamp(BACKTEST_START)
    end_ts = _latest_feature_date()
    if end_ts is None or pd.isna(end_ts):
        return pd.DataFrame(columns=["ts","equity","daily_return","drawdown"])

    feats = _load_features_window(start_ts, end_ts)
    if feats.empty:
        return pd.DataFrame(columns=["ts","equity","daily_return","drawdown"])

    px = _load_prices_window(start_ts, end_ts + pd.Timedelta(days=TARGET_HORIZON_DAYS+1))
    if px.empty:
        return pd.DataFrame(columns=["ts","equity","daily_return","drawdown"])

    # forward returns
    px = px.sort_values(["symbol","ts"]).copy()
    px["px_fwd"] = px.groupby("symbol")["px"].shift(-TARGET_HORIZON_DAYS)
    px["fwd_ret"] = (px["px_fwd"] / px["px"]) - 1.0

    dates = sorted(feats["ts"].unique())
    if len(dates) < 40:
        return pd.DataFrame(columns=["ts","equity","daily_return","drawdown"])

    equity = 1.0
    rows = []
    cur_month = None
    pipe = None

    for d in dates[:-TARGET_HORIZON_DAYS]:
        d = pd.Timestamp(d)
        if cur_month != d.month:
            # train using last `window_years` years of data strictly before d
            train_start = max(start_ts, d - pd.DateOffset(years=window_years))
            joined = feats.merge(px[["symbol","ts","fwd_ret"]], on=["symbol","ts"], how="left")
            train_df = joined[(joined["ts"] < d) & (joined["ts"] >= train_start) & (~joined["fwd_ret"].isna())]
            if not train_df.empty:
                X = train_df[FEATURE_COLS].values
                y = train_df["fwd_ret"].values
                pipe = Pipeline([("scaler", StandardScaler()), ("xgb", XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=0, tree_method="hist"))])
                pipe.fit(X, y)
            cur_month = d.month

        if pipe is None:
            continue
        today_feats = feats.loc[feats["ts"] == d]
        if today_feats.empty:
            continue
        preds = pipe.predict(today_feats[FEATURE_COLS].values)
        top = today_feats.assign(pred=preds).sort_values("pred", ascending=False).head(TOP_N)

        realized = px[(px["symbol"].isin(top["symbol"])) & (px["ts"] == d)]["fwd_ret"].mean()
        if pd.isna(realized):
            continue
        slip = (SLIPPAGE_BPS / 10000.0) * 2.0
        equity *= (1.0 + realized - slip)
        rows.append({"ts": d.date(), "equity": equity})

    if not rows:
        return pd.DataFrame(columns=["ts","equity","daily_return","drawdown"])
    df = pd.DataFrame(rows).sort_values("ts")
    df["daily_return"] = df["equity"].pct_change().fillna(0.0)
    rolling_max = df["equity"].cummax()
    df["drawdown"] = df["equity"] / rolling_max - 1.0
    upsert_dataframe(df, BacktestEquity, ["ts"])
    return df
