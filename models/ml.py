from __future__ import annotations
import numpy as np, pandas as pd, copy, logging, os
from os import cpu_count
from typing import Dict, Any
from sqlalchemy import text, inspect
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
from utils.price_utils import price_expr
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor  # ensure xgboost is in your requirements
log = logging.getLogger(__name__)

# Feature set used for model training and prediction.  Extended to include
# volatility-adjusted reversal and idiosyncratic volatility.  The list is
# intentionally ordered so that price/momentum/volatility features come first,
# followed by liquidity/size/beta and additional features.  Sparse event
# features (like PEAD and Russell index inclusion) are appended at the end.
FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_21d", "mom_21", "mom_63", "vol_21", "rsi_14",
    "turnover_21", "size_ln", "overnight_gap", "illiq_21", "beta_63",
    # Volatility-adjusted reversal and idiosyncratic volatility
    "reversal_5d_z", "ivol_63",
    # New technical indicators
    "ema_12", "ema_26", "macd", "ema_50", "ema_200", "ma_ratio_50_200",
    "vol_63", "vol_252", "mom_252",
    "spread_ratio", "spread_21", "atr_14", "obv",
    # Fundamental ratios
    "f_pe_ttm", "f_pb", "f_ps_ttm", "f_debt_to_equity", "f_roa", "f_gm",
    "f_profit_margin", "f_current_ratio",
    # Sparse high-impact event features
    "pead_event", "pead_surprise_eps", "pead_surprise_rev", "russell_inout"
]
TCA_COLS = ["vol_21", "adv_usd_21"]

# Target variable used for training and evaluation.  By default, we model
# the residualized forward return (fwd_ret_resid) which removes the
# component explained by the stock's beta to the market.  To use the raw
# forward return instead, change this constant to 'fwd_ret'.
TARGET_VARIABLE = 'fwd_ret_resid'

# Coverage fallback configuration
MIN_TARGET_COVERAGE = float(os.getenv("MIN_TARGET_COVERAGE", "0.6"))  # 60% minimum coverage
FALLBACK_TARGET = 'fwd_ret'  # Fallback target when primary coverage is low

def _latest_feature_date():
    try:
        df = pd.read_sql_query(text("SELECT MAX(ts) AS max_ts FROM features"), engine, parse_dates=["max_ts"])
        return df.iloc[0,0]
    except Exception:
        return None

def _load_features_window(start_ts, end_ts):
    """
    Load a window of features between start_ts and end_ts, inclusive.

    The function selects all feature columns, trading cost adjustment columns,
    and the target variable from the `features` table.  If any column is
    missing from the table, it will be absent in the resulting DataFrame.
    """
    # Always include identifier columns, the feature columns, TCA columns and the target
    cols = list(set(FEATURE_COLS + TCA_COLS + ["symbol", "ts", TARGET_VARIABLE, 'fwd_ret']))
    # Determine which columns actually exist in the database
    inspector = inspect(engine)
    existing_cols = {c['name'] for c in inspector.get_columns('features')}
    missing = [c for c in cols if c not in existing_cols]
    if missing:
        log.warning("Missing feature columns: %s", missing)
    cols_to_query = [c for c in cols if c in existing_cols]
    if cols_to_query:
        cols_str = ", ".join(cols_to_query)
    else:
        # Fallback: only use identifier columns if they exist
        fallback_cols = [c for c in ["symbol", "ts"] if c in existing_cols]
        if fallback_cols:
            cols_str = ", ".join(fallback_cols)
        else:
            log.error("No columns found in 'features' table to query.")
            return pd.DataFrame(columns=cols)
    sql = f"SELECT {cols_str} FROM features WHERE ts >= :start AND ts <= :end"
    try:
        df = pd.read_sql_query(text(sql), engine,
                               params={"start": start_ts.date(), "end": end_ts.date()},
                               parse_dates=["ts"])
        for col in missing:
            df[col] = np.nan
        # ensure deterministic order and column set
        df = df.reindex(columns=cols)
        return df.sort_values(["ts", "symbol"]) if not df.empty else df
    except Exception as e:
        log.error(f"Failed to load features window: {e}")
        return pd.DataFrame(columns=cols)

def _load_prices_window(start_ts, end_ts):
    try:
        return pd.read_sql_query(text(f"SELECT symbol, ts, {price_expr()} AS px FROM daily_bars WHERE ts>=:s AND ts<=:e"),
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
    """
    Define the base model specifications to be blended.

    Returns a dictionary mapping model names to sklearn Pipelines.  In addition
    to standard regression models (RandomForest, Ridge, XGBoost), this
    function conditionally includes a learning-to-rank (LTR) variant of
    XGBoost if the XGBoost library is installed.  The LTR model uses
    the 'rank:ndcg' objective and will be fit with group information.
    """
    specs: Dict[str, Pipeline] = {
        "rf": _define_pipeline(RandomForestRegressor(n_estimators=350, max_depth=10,
                                                     random_state=42, n_jobs=_xgb_threads())),
        "ridge": _define_pipeline(Ridge(alpha=1.0)),
    }
    if XGBRegressor:
        # Standard regression XGBoost model
        specs["xgb_reg"] = _define_pipeline(XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=_xgb_threads(),
            tree_method="hist"
        ))
        # Learning-to-rank model using NDCG objective
        specs["xgb_ltr"] = _define_pipeline(XGBRegressor(
            objective='rank:ndcg',
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=_xgb_threads(),
            tree_method="hist"
        ))
    return specs

def _parse_blend_weights(s: str) -> Dict[str, float]:
    out = {}
    for part in s.split(","):
        if ":" in part:
            k,v = part.split(":",1)
            try: out[k.strip()] = float(v)
            except: pass
    t = sum(out.values()) or 1.0
    return {k: v/t for k,v in out.items()}

def _calculate_target_coverage(df: pd.DataFrame, target_col: str) -> float:
    """
    Calculate coverage rate for a target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the target column
    target_col : str
        Name of the target column to check coverage for
        
    Returns
    -------
    float
        Coverage rate between 0.0 and 1.0
    """
    if df.empty or target_col not in df.columns:
        return 0.0
    
    total_rows = len(df)
    non_null_rows = df[target_col].notna().sum()
    
    return non_null_rows / total_rows if total_rows > 0 else 0.0

def train_with_cv(X_train, y_train, base_model, param_grid):
    """Train a model using time‑series cross‑validation and return the best estimator."""
    if param_grid:
        cv = TimeSeriesSplit(n_splits=5)
        search = GridSearchCV(base_model, param_grid, cv=cv,
                              scoring='neg_mean_absolute_error', n_jobs=-1)
        search.fit(X_train, y_train)
        return search.best_estimator_
    else:
        return base_model.fit(X_train, y_train)

def _select_target_with_fallback(df: pd.DataFrame, primary_target: str, fallback_target: str, min_coverage: float) -> str:
    """
    Select the best target variable based on coverage thresholds.
    
    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame
    primary_target : str
        Preferred target variable (e.g., 'fwd_ret_resid')
    fallback_target : str
        Fallback target variable (e.g., 'fwd_ret')
    min_coverage : float
        Minimum coverage threshold (0.0 to 1.0)
        
    Returns
    -------
    str
        Selected target variable name
    """
    primary_coverage = _calculate_target_coverage(df, primary_target)
    fallback_coverage = _calculate_target_coverage(df, fallback_target)
    
    log.info(f"Target coverage analysis:")
    log.info(f"  {primary_target}: {primary_coverage:.1%}")
    log.info(f"  {fallback_target}: {fallback_coverage:.1%}")
    log.info(f"  Minimum threshold: {min_coverage:.1%}")
    
    if primary_coverage >= min_coverage:
        log.info(f"Using primary target: {primary_target}")
        return primary_target
    elif fallback_coverage >= min_coverage:
        log.warning(f"Primary target coverage too low, falling back to: {fallback_target}")
        return fallback_target
    else:
        log.warning(f"Both targets have low coverage, using primary anyway: {primary_target}")
        return primary_target

def _ic_by_model(train_df: pd.DataFrame, feature_cols: list[str], target_variable: str = TARGET_VARIABLE) -> Dict[str,float]:
    """
    Compute information coefficients (IC) for each model over a recent evaluation window.

    Parameters
    ----------
    train_df : DataFrame
        Training data containing features, the target variable, and timestamps.
    feature_cols : list[str]
        List of feature column names to be used as inputs to the models.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping model names to their normalized (positive) mean ICs.

    Notes
    -----
    - IC is computed as the mean Spearman correlation between model predictions and
      the target variable across dates in the recent 6-month window.  If there
      are insufficient observations in the window, the full training set is used.
    - Learning-to-rank models receive a `group` parameter corresponding to the
      number of samples per date.
    - Negative ICs are floored at zero before normalization.
    """
    models = _model_specs()
    if train_df.empty:
        return {}
    ics: Dict[str, float] = {}
    # Define evaluation window: last ~6 months
    recent_mask = train_df['ts'] >= (train_df['ts'].max() - pd.Timedelta(days=180))
    Xr = train_df.loc[recent_mask, ['ts'] + feature_cols]
    yr = train_df.loc[recent_mask, target_variable]
    if Xr.empty:
        # fallback to full data
        Xr = train_df[['ts'] + feature_cols]
        yr = train_df[target_variable]
    # Prepare full training matrices (used for fitting models)
    X_train = train_df[['ts'] + feature_cols]
    y_train = train_df[target_variable]  # Keep as Series for proper indexing
    # For LTR models, compute group sizes: sorted by ts
    sorted_idx = X_train.sort_values('ts').index
    group_sizes = X_train.loc[sorted_idx].groupby('ts').size().values
    # Evaluate each model
    for name, pipe in models.items():
        try:
            p2 = copy.deepcopy(pipe)
            fit_params = {}
            # For LTR models (identified by 'ltr' in name), pass group parameter
            if 'ltr' in name:
                fit_params['model__group'] = group_sizes
            # Fit on full training set (sorted by timestamp for LTR)
            if len(sorted_idx) == len(X_train):
                X_train_sorted = X_train.loc[sorted_idx, feature_cols]
                y_train_sorted = y_train.loc[sorted_idx].values  # Use .loc for label-based indexing, then convert to array
            else:
                X_train_sorted = X_train[feature_cols]
                y_train_sorted = y_train.values
            p2.fit(X_train_sorted, y_train_sorted, **fit_params)
            # Predict on evaluation window
            X_eval = Xr[feature_cols]
            preds = pd.Series(p2.predict(X_eval), index=Xr.index)
            # Compute Spearman IC by date
            tmp = pd.DataFrame({
                'ts': Xr['ts'],
                'pred': preds.values,
                'y': yr.values
            })
            ic_by_date = tmp.groupby('ts').apply(lambda d: d['pred'].corr(d['y'], method='spearman'))
            ics[name] = float(ic_by_date.fillna(0.0).mean())
        except Exception as e:
            log.error(f"Failed to calculate IC for model {name}: {e}")
            ics[name] = 0.0
    # Normalize: floor negative ICs at zero and scale to sum to one
    pos = {k: max(0.0, v) for k, v in ics.items()}
    s = sum(pos.values()) or 1.0
    return {k: v/s for k, v in pos.items()}

def train_and_predict_all_models(window_years: int = 4):
    log.info("Starting live training and prediction (v17).")
    latest_ts = _latest_feature_date()
    if latest_ts is None or pd.isna(latest_ts):
        log.warning("No features available. Cannot train models.")
        return {}

    start_ts = max(pd.Timestamp(BACKTEST_START), latest_ts - pd.DateOffset(years=window_years))
    feats = _load_features_window(start_ts, latest_ts)
    if feats.empty:
        return {}

    # Merge sparse AltSignals (if any)
    alts = _load_altsignals(start_ts, latest_ts)
    if not alts.empty:
        feats = feats.merge(alts, on=['symbol', 'ts'], how='left')

    # Survivorship gating using historical universe snapshots
    try:
        feats = gate_training_with_universe(feats)
    except Exception as e:
        log.info(f"Universe gating skipped: {e}")

    # Select target variable with coverage fallback
    selected_target = _select_target_with_fallback(
        feats, TARGET_VARIABLE, FALLBACK_TARGET, MIN_TARGET_COVERAGE
    )
    log.info(f"Selected target variable: {selected_target}")

    # Identify training and latest sets
    train_df = feats.dropna(subset=[selected_target]).copy()
    latest_df = feats[feats['ts'] == latest_ts].copy()
    if train_df.empty or latest_df.empty:
        log.warning("Training or prediction sets are empty.")
        return {}

    # Fill missing feature columns with zeros (for sparse features)
    for c in FEATURE_COLS:
        if c not in train_df.columns:
            train_df[c] = 0.0
        if c not in latest_df.columns:
            latest_df[c] = 0.0

    # After ensuring all feature columns exist, fill missing values.
    # This prevents None/NaN from reaching the model.
    train_df[FEATURE_COLS] = train_df[FEATURE_COLS].fillna(0.0)
    latest_df[FEATURE_COLS] = latest_df[FEATURE_COLS].fillna(0.0)

    # Prepare training matrix and target
    ID_COLS = ['ts', 'symbol']
    X = train_df[ID_COLS + FEATURE_COLS]
    y = train_df[selected_target].values
    X_latest = latest_df[FEATURE_COLS]

    # Apply time-decay sample weights (half-life = 120 days)
    HALF_LIFE_DAYS = 120
    decay_factor = np.log(2) / HALF_LIFE_DAYS
    latest_train_ts = X['ts'].max()
    sample_age_days = (latest_train_ts - X['ts']).dt.days
    sample_weights = np.exp(-decay_factor * sample_age_days)
    # Normalize weights: sum to number of samples (prevents small values)
    sample_weights = (sample_weights / sample_weights.sum() * len(sample_weights))
    sample_weights.index = X.index  # ensure alignment
    log.info(f"Applied time-decay sample weights (Half-life={HALF_LIFE_DAYS}d).")

    # For LTR models, X must be sorted by ts and group sizes specified
    # Sort training data by ts to compute group sizes
    if not X.index.equals(X.sort_values('ts').index):
        sorted_idx = X.sort_values('ts').index
        X_train_sorted = X.loc[sorted_idx, FEATURE_COLS]
        y_train_sorted = train_df.loc[sorted_idx, selected_target].values
        sample_weights = sample_weights.loc[sorted_idx]
    else:
        sorted_idx = X.index
        X_train_sorted = X[FEATURE_COLS]
        y_train_sorted = y

    # Group sizes: number of samples per date in sorted training set
    group_sizes = X.loc[sorted_idx].groupby('ts').size().values

    models = _model_specs()
    outputs: Dict[str, Any] = {}
    preds_dict: Dict[str, pd.Series] = {}

    # Compute blend weights from config and adaptive IC weighting
    from config import BLEND_WEIGHTS, REGIME_GATING
    blend_w = _parse_blend_weights(BLEND_WEIGHTS)
    ic_w = _ic_by_model(train_df[['ts','symbol', selected_target] + FEATURE_COLS], FEATURE_COLS, selected_target)
    if ic_w:
        keys = set(blend_w) | set(ic_w)
        combined = {k: 0.5 * blend_w.get(k, 0) + 0.5 * ic_w.get(k, 0) for k in keys}
        s = sum(combined.values()) or 1.0
        blend_w = {k: v / s for k, v in combined.items()}

    # Regime gating (optional)
    if REGIME_GATING:
        regime = classify_regime(latest_ts)
        blend_w = gate_blend_weights(blend_w, regime)
        log.info(f"Regime: {regime}; Blend weights gated: {blend_w}")

    # Fit & predict base models
    for name, (model, grid) in _model_specs().items():
    best_model = train_with_cv(X_train, y_train, model, grid)
    for name, pipe in models.items():
        p2 = copy.deepcopy(pipe)
        # Prepare fit parameters
        fit_params: Dict[str, Any] = {}
        # Sample weights (if supported)
        fit_params['model__sample_weight'] = sample_weights.values
        # Group sizes for LTR models
        if 'ltr' in name:
            fit_params['model__group'] = group_sizes
        try:
            # Fit model
            p2.fit(X_train_sorted, y_train_sorted, **fit_params)
        except TypeError as e:
            # Some models may not support sample_weight; retry without it
            if 'sample_weight' in str(e) and 'model__sample_weight' in fit_params:
                log.warning(f"Model {name} does not support sample_weight. Training without weights.")
                fit_params.pop('model__sample_weight')
                p2.fit(X_train_sorted, y_train_sorted, **fit_params)
            else:
                log.error(f"Error during fitting model {name}: {e}")
                continue
        except Exception as e:
            log.error(f"Model training failed for {name}: {e}")
            continue
        # Predict on latest set
        try:
            pred_vals = p2.predict(X_latest)
        except Exception as e:
            log.error(f"Prediction failed for model {name}: {e}")
            continue
        out = latest_df[['symbol']].copy()
        out['ts'] = latest_ts
        out['y_pred'] = pred_vals.astype(float)
        out['model_version'] = f"{name}_v1"
        preds_dict[name] = out.set_index('symbol')['y_pred']
        outputs[name] = out.copy()
        # Persist predictions
        upsert_dataframe(out[['symbol','ts','y_pred','model_version']], Prediction, ['symbol','ts','model_version'])

    # Create blended predictions (raw and neutralized)
    if blend_w:
        sym_index = latest_df['symbol']
        blend = np.zeros(len(sym_index))
        for name, w in blend_w.items():
            if name in preds_dict:
                series = preds_dict[name].reindex(sym_index).fillna(0.0).values
                blend += w * series
        # Raw blend output
        out = latest_df[['symbol']].copy()
        out['ts'] = latest_ts
        out['y_pred'] = blend.astype(float)
        out['model_version'] = 'blend_raw_v1'
        outputs['blend_raw'] = out.copy()
        upsert_dataframe(out[['symbol','ts','y_pred','model_version']], Prediction, ['symbol','ts','model_version'])
        # Sector/factor neutralization
        try:
            # Factor exposures used for neutralization; fallback to just size_ln, mom_21, turnover_21
            fac_cols = ['size_ln', 'mom_21', 'turnover_21']
            if 'beta_63' in latest_df.columns:
                fac_cols.append('beta_63')
            fac = latest_df.set_index('symbol')[fac_cols]
            pred_series = pd.Series(blend, index=sym_index)
            sd = build_sector_dummies(sym_index.tolist(), latest_ts)
            resid = neutralize_with_sectors(pred_series, fac, sd)
            out2 = latest_df[['symbol']].copy()
            out2['ts'] = latest_ts
            out2['y_pred'] = resid.values
            out2['model_version'] = 'blend_v1'
            outputs['blend'] = out2.copy()
            upsert_dataframe(out2[['symbol','ts','y_pred','model_version']], Prediction, ['symbol','ts','model_version'])
        except Exception as e:
            log.warning(f"Neutralization failed: {e}; fallback to raw blend.")
            out_fallback = out.copy()
            out_fallback['model_version'] = 'blend_v1'
            upsert_dataframe(out_fallback[['symbol','ts','y_pred','model_version']], Prediction, ['symbol','ts','model_version'])
    log.info("Live training and prediction complete.")
    return outputs

def _model_specs():
    return {
        "ridge": (Ridge(), {"alpha": [0.1, 1.0, 10.0]}),
        "random_forest": (RandomForestRegressor(), {"n_estimators": [200, 400], "max_depth": [3, 5, None]}),
        "xgb": (XGBRegressor(objective="reg:squarederror"), {
            "n_estimators": [200, 300],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0]
        })
    }

# --- Minimal walk-forward backtest (used by Streamlit app) ---
from config import PREFERRED_MODEL
from db import BacktestEquity

def run_walkforward_backtest(model_version: str | None = None, top_n: int = 20) -> pd.DataFrame:
    """
    Simple WFB: for each ts, take top-N positive names by y_pred (given model_version),
    realize fwd_ret over TARGET_HORIZON_DAYS, average cross-sectionally, and cumulate to equity.
    Persists results to backtest_equity.
    """
    mv = model_version or PREFERRED_MODEL
    with engine.connect() as con:
        preds = pd.read_sql_query(
            text("SELECT symbol, ts, y_pred FROM predictions WHERE model_version = :mv ORDER BY ts, y_pred DESC"),
            con, params={"mv": mv}, parse_dates=['ts']
        )
    if preds.empty:
        log.warning("No predictions found for backtest.")
        return pd.DataFrame(columns=['ts','equity','daily_return','drawdown','tcost_impact'])
    # Compute realized forward returns
    with engine.connect() as con:
        px = pd.read_sql_query(
            text(f"SELECT symbol, ts, {price_expr()} AS px FROM daily_bars"),
            con, parse_dates=['ts']
        )
    if px.empty:
        log.warning("No prices for backtest.")
        return pd.DataFrame(columns=['ts','equity','daily_return','drawdown','tcost_impact'])
    px = px.sort_values(['symbol','ts'])
    px['px_fwd'] = px.groupby('symbol')['px'].shift(-TARGET_HORIZON_DAYS)
    px['fwd_ret'] = (px['px_fwd']/px['px']) - 1.0
    df = preds.merge(px[['symbol','ts','fwd_ret']], on=['symbol','ts'], how='left')
    if df['fwd_ret'].isna().all():
        log.warning("No fwd returns matched for backtest.")
        return pd.DataFrame(columns=['ts','equity','daily_return','drawdown','tcost_impact'])
    # Pick top-N positive per ts
    port = (
        df[df['y_pred'] > 0]
        .sort_values(['ts','y_pred'], ascending=[True, False])
        .groupby('ts')
        .head(top_n)
        .groupby('ts')['fwd_ret'].mean()
        .dropna()
        .sort_index()
    )
    if port.empty:
        return pd.DataFrame(columns=['ts','equity','daily_return','drawdown','tcost_impact'])
    equity = (1.0 + port).cumprod()
    dd = equity / equity.cummax() - 1.0
    out = pd.DataFrame({'ts': port.index.normalize(), 'equity': equity.values, 'daily_return': port.values, 'drawdown': dd.values, 'tcost_impact': 0.0})
    # Persist
    upsert_dataframe(out, BacktestEquity, ['ts'])
    return out
