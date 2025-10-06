from __future__ import annotations

import numpy as np, pandas as pd, copy, logging, os
from os import cpu_count
from typing import Dict, Any
from sqlalchemy import text, inspect
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor

# Optional third‑party regressors.  Not all environments (e.g. Render
# workers) have LightGBM or CatBoost installed.  Import these in try/except
# blocks so the worker can start even when the dependencies are missing.
try:
    from xgboost import XGBRegressor, XGBRanker  # type: ignore[attr-defined]
except Exception:
    # If xgboost isn't available, set both classes to None.  Note: the
    # indentation here is critical.  Both assignments must be at the same
    # indentation level inside the except block to avoid an IndentationError.
    XGBRegressor = None  # type: ignore[assignment]
    XGBRanker = None  # type: ignore[assignment]

try:
    from lightgbm import LGBMRegressor  # type: ignore[attr-defined]
except Exception:
    LGBMRegressor = None  # type: ignore[assignment]

try:
    from catboost import CatBoostRegressor  # type: ignore[attr-defined]
except Exception:
    CatBoostRegressor = None  # type: ignore[assignment]

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from db import engine, upsert_dataframe, Prediction
from utils.prediction_metadata import as_naive_utc, with_prediction_metadata
from models.transformers import CrossSectionalNormalizer
from models.rl_models import QTableRegressor
from config import (BACKTEST_START, TARGET_HORIZON_DAYS, BLEND_WEIGHTS,
                    SECTOR_NEUTRALIZE,)

# Optional regime‑gating configuration.  If USE_REGIME_GATING is set to
# 'true', the pipeline will compute a volatility‑driven gating factor for
# the latest prediction date and blend tree‑based and neural models based
# on this factor.  GATING_THRESHOLD defines the reference 21‑day
# volatility (e.g. median or long‑run average) and GATING_SLOPE controls
# how rapidly the gating transitions between 0 and 1.  These can be set via
# environment variables or default to reasonable values.
USE_REGIME_GATING = os.getenv('USE_REGIME_GATING', 'false').lower() == 'true'
try:
    GATING_THRESHOLD = float(os.getenv('GATING_THRESHOLD', '0.03'))  # 3% daily vol
except Exception:
    GATING_THRESHOLD = 0.03
try:
    GATING_SLOPE = float(os.getenv('GATING_SLOPE', '5.0'))
except Exception:
    GATING_SLOPE = 5.0

# -----------------------------------------------------------------------------
# Hyper-parameter search configuration.  By default the pipeline uses
# GridSearchCV for deterministic hyper-parameter tuning.  If
# USE_RANDOM_SEARCH is set to 'true' in the environment, the training
# routine will instead use RandomizedSearchCV with a limited number
# of iterations.  This can speed up the search and explore more
# parameter combinations.  The number of random iterations can be
# configured via N_RANDOM_SEARCH_ITER.
USE_RANDOM_SEARCH = os.getenv('USE_RANDOM_SEARCH', 'false').lower() == 'true'
try:
    N_RANDOM_SEARCH_ITER = int(os.getenv('N_RANDOM_SEARCH_ITER', '10'))
except Exception:
    N_RANDOM_SEARCH_ITER = 10
# Halving random search (successive halving) configuration.  When enabled,
# the training routine will use HalvingRandomSearchCV to perform adaptive
# hyper-parameter search.  This often yields better performance than a
# fixed‑size random search by allocating more resources to promising
# configurations.  Set USE_HALVING_SEARCH=true in the environment to
# activate.  You can optionally override the halving factor via
# HALVING_FACTOR.  See sklearn.experimental HalvingRandomSearchCV docs.
USE_HALVING_SEARCH = os.getenv('USE_HALVING_SEARCH', 'false').lower() == 'true'
try:
    HALVING_FACTOR = float(os.getenv('HALVING_FACTOR', '3.0'))
except Exception:
    HALVING_FACTOR = 3.0
from risk.sector import build_sector_dummies
from risk.risk_model import neutralize_with_sectors
from models.regime import classify_regime, gate_blend_weights
from data.universe_history import gate_training_with_universe
from utils.price_utils import price_expr
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
# Import successive halving search (experimental).  This must be imported
# via sklearn.experimental to enable the classes.  We alias the import
# for potential use in hyper‑parameter tuning when USE_HALVING_SEARCH is enabled.
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
import logging

log = logging.getLogger(__name__)

# Feature set used for model training and prediction.  Trimmed to include only
# features that are actually computed by models/features.py.  The list is
# intentionally ordered so that price/momentum/volatility features come first,
# followed by liquidity/size/beta, fundamental ratios, macro features and sparse
# event indicators.  See models/features.py for generation of these features.
FEATURE_COLS = [
    # Price/returns and momentum
    "ret_1d", "ret_5d", "ret_21d",
    "mom_21", "mom_63",
    # Volatility, RSI and liquidity
    "vol_21", "rsi_14",
    "turnover_21",
    "size_ln",
    "overnight_gap",
    "illiq_21",
    "beta_63",
    # Liquidity measure used in cross-sectional z‑scores
    "adv_usd_21",
    # Fundamental ratios
    "f_pe_ttm", "f_pb", "f_ps_ttm", "f_debt_to_equity",
    "f_roa", "f_roe", "f_gross_margin", "f_profit_margin", "f_current_ratio",
    # Market‑level macro features (from benchmark returns) across multiple horizons
    "mkt_ret_1d", "mkt_ret_5d", "mkt_ret_21d", "mkt_ret_63d",
    "mkt_vol_21", "mkt_vol_63",
    "mkt_skew_21", "mkt_kurt_21", "mkt_skew_63", "mkt_kurt_63",
    # Sparse high-impact event features
    "pead_event", "pead_surprise_eps", "pead_surprise_rev", "russell_inout",
]

# Cross‑sectional z‑score features.  These columns (prefixed with ``cs_z_``)
# are generated by the feature engineering pipeline to capture each stock’s
# relative standing within the universe on a given day.  If these
# columns exist in the features table, they will be included in the
# modelling; otherwise they will be silently ignored when loading data.
CS_Z_FEATURES = [
    'mom_21', 'mom_63', 'vol_21', 'rsi_14', 'turnover_21', 'size_ln',
    'adv_usd_21', 'beta_63',
    'f_pe_ttm', 'f_pb', 'f_ps_ttm', 'f_debt_to_equity',
    'f_roa', 'f_roe', 'f_gross_margin', 'f_profit_margin', 'f_current_ratio'
]
# Append the z‑score columns to the feature list.  We do this outside the
# literal definition to reflect that these features are optional; if not
# present in the DB they will be backfilled with NaN and later replaced
# with zeros during training.
for _base in CS_Z_FEATURES:
    FEATURE_COLS.append(f'cs_z_{_base}')

# Trading cost adjustment columns
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
        return df.iloc[0, 0]
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
        df = pd.read_sql_query(
            text(sql),
            engine,
            params={"start": start_ts.date(), "end": end_ts.date()},
            parse_dates=["ts"],
        )
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
        return pd.read_sql_query(
            text(f"SELECT symbol, ts, {price_expr()} AS px FROM daily_bars WHERE ts>=:s AND ts<=:e"),
            engine,
            params={'s': start_ts.date(), 'e': end_ts.date()},
            parse_dates=['ts']
        ).sort_values(['symbol', 'ts'])
    except Exception:
        return pd.DataFrame(columns=['symbol', 'ts', 'px'])


def _load_altsignals(start_ts, end_ts):
    try:
        df = pd.read_sql_query(
            text("SELECT symbol, ts, name, value FROM alt_signals WHERE ts>=:s AND ts<=:e"),
            engine,
            params={'s': start_ts.date(), 'e': end_ts.date()},
            parse_dates=['ts']
        )
    except Exception:
        return pd.DataFrame(columns=['symbol', 'ts', 'name', 'value'])
    if df.empty:
        return df
    # pivot sparse signals
    piv = df.pivot_table(index=['symbol', 'ts'], columns='name', values='value', aggfunc='last').reset_index()
    return piv


def _xgb_threads() -> int:
    c = cpu_count() or 2
    return max(1, c - 1)


def _define_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([
        ("normalizer", CrossSectionalNormalizer(winsorize_tails=0.05)),
        ("model", estimator),
    ])


def _model_specs() -> Dict[str, Pipeline]:
    """
    Define the base model specifications to be blended.

    Returns a dictionary mapping model names to sklearn Pipelines.  In addition
    to standard regression models (RandomForest, Ridge, XGBoost), this
    function conditionally includes a learning-to-rank (LTR) variant of
    XGBoost if the XGBoost library is installed.  The LTR model always falls
    back to a regression objective (rmse) using XGBRegressor to avoid
    integer-label constraints.
    """
    specs: Dict[str, Pipeline] = {
        "rf": _define_pipeline(RandomForestRegressor(
            n_estimators=350,
            max_depth=10,
            random_state=42,
            n_jobs=_xgb_threads()
        )),
        "ridge": _define_pipeline(Ridge(alpha=1.0)),
        # ExtraTrees model: extremely randomized trees for robust non-linear patterns
        "extra_trees": _define_pipeline(ExtraTreesRegressor(
            n_estimators=400,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=False,
            random_state=42,
            n_jobs=_xgb_threads()
        )),
        # Gradient Boosting Regressor: additive tree boosting (lighter alternative to XGB)
        "gbr": _define_pipeline(GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42
        )),
        # ElasticNet linear model: combines L1 and L2 regularization; can capture sparse relationships
        "elasticnet": _define_pipeline(ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)),
    }
    # Only include XGBoost models if the import succeeded.  The
    # presence of XGBRegressor is used as a proxy for all optional models,
    # since LightGBM and CatBoost are also imported conditionally.  This
    # prevents NameError on workers where these libraries are absent.
    if XGBRegressor is not None:
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
        # Always define the LTR model using XGBRegressor with rmse objective.  Using
        # XGBRanker triggers integer-label constraints which are not met here.
        specs["xgb_ltr"] = _define_pipeline(XGBRegressor(
            objective='rmse',
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=_xgb_threads(),
            tree_method="hist"
        ))
    # LightGBM regression model (if available)
    if XGBRegressor is None or LGBMRegressor is not None:
        # include LightGBM even if xgboost is absent; unify with existing spec
        if LGBMRegressor is not None:
            specs["lgbm"] = _define_pipeline(LGBMRegressor(
                n_estimators=500,
                num_leaves=31,
                max_depth=-1,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=_xgb_threads()
            ))
    # CatBoost regression model (if available).  Set verbose=0 to silence output.
    if CatBoostRegressor is not None:
        specs["cat"] = _define_pipeline(CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function='RMSE',
            random_seed=42,
            verbose=False
        ))
    # Stacking ensemble combining tree-based models. Use base estimators directly in the stacker.
    stack_estimators = []
    if XGBRegressor is not None:
        stack_estimators.append(("xgb", XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=_xgb_threads(),
            tree_method="hist"
        )))
    if LGBMRegressor is not None:
        stack_estimators.append(("lgbm", LGBMRegressor(
            n_estimators=500,
            num_leaves=31,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=_xgb_threads()
        )))
    if CatBoostRegressor is not None:
        stack_estimators.append(("cat", CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function='RMSE',
            random_seed=42,
            verbose=False
        )))
    # Create the stacking model only if we have at least two base estimators
    if len(stack_estimators) >= 2:
        stack_model = StackingRegressor(
            estimators=stack_estimators,
            final_estimator=Ridge(alpha=1.0),
            n_jobs=_xgb_threads(),
            passthrough=False
        )
        specs["stack"] = _define_pipeline(stack_model)
    # Feedforward neural network (MLP) model
    specs["mlp"] = _define_pipeline(MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    ))
    # Deep feedforward neural network with three hidden layers.
    specs["deep_mlp"] = _define_pipeline(MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=800,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    ))
    # HistGradientBoostingRegressor: a fast, tree‑based gradient boosting
    # implementation that can approximate deeper tree interactions.
    try:
        _ = HistGradientBoostingRegressor
        specs["hgb"] = _define_pipeline(HistGradientBoostingRegressor(
            loss='lsquared_error',
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=0.0,
            random_state=42
        ))
    except Exception:
        pass
    # Very deep feedforward neural network with four hidden layers.
    specs["very_deep_mlp"] = _define_pipeline(MLPRegressor(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    ))
    # Simple reinforcement-learning inspired Q-table model
    specs["q_table"] = _define_pipeline(QTableRegressor(n_bins=10, random_state=42))
    return specs

# Hyper‑parameter grids for each model.  These grids are designed to be
# relatively small to keep cross‑validation feasible.  Keys should match
# entries in the models dictionary returned by `_model_specs`.
# Parameter names must include the 'model__' prefix to refer to the estimator
# inside the Pipeline.  Models absent from this mapping will be fit
# without tuning.
PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    'rf': {
        'model__n_estimators': [200, 400, 600],
        'model__max_depth': [8, 12, None],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'extra_trees': {
        'model__n_estimators': [300, 500],
        'model__max_depth': [8, 12, None],
        'model__min_samples_split': [2, 4],
        'model__min_samples_leaf': [1, 2]
    },
    'gbr': {
        'model__n_estimators': [200, 400],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [2, 3],
        'model__subsample': [0.6, 0.8]
    },
    'elasticnet': {
        'model__alpha': [0.01, 0.1, 0.5],
        'model__l1_ratio': [0.2, 0.5, 0.8]
    },
    'xgb_reg': {
        'model__n_estimators': [400, 600],
        'model__max_depth': [3, 4, 5],
        'model__learning_rate': [0.03, 0.1],
        'model__subsample': [0.6, 0.8],
        'model__colsample_bytree': [0.6, 0.8]
    },
    'lgbm': {
        'model__n_estimators': [400, 600],
        'model__num_leaves': [31, 63],
        'model__learning_rate': [0.03, 0.1],
        'model__subsample': [0.6, 0.8],
        'model__colsample_bytree': [0.6, 0.8]
    },
    'cat': {
        'model__iterations': [200, 400],
        'model__depth': [4, 6],
        'model__learning_rate': [0.03, 0.1]
    },
    'mlp': {
        'model__hidden_layer_sizes': [(128, 64), (256, 128)],
        'model__alpha': [1e-4, 1e-3],
        'model__learning_rate_init': [1e-3, 5e-3]
    },
    # Deeper MLP parameter grid.  We test different layer configurations
    # and regularization strengths.  The deeper network may capture
    # higher‑order interactions but risks overfitting; cross‑validation
    # chooses the best configuration.
    'deep_mlp': {
        'model__hidden_layer_sizes': [(256, 128, 64), (512, 256, 128)],
        'model__alpha': [1e-4, 1e-3, 1e-2],
        'model__learning_rate_init': [1e-3, 5e-3]
    },
    # HistGradientBoostingRegressor parameter grid.  Tune learning rate,
    # maximum tree depth and number of iterations.  The library handles
    # missing values automatically so we don't need to set min_samples.
    'hgb': {
        'model__learning_rate': [0.03, 0.1],
        'model__max_depth': [4, 6, 8],
        'model__max_iter': [200, 400]
    },
    # Very deep MLP parameter grid: experiment with more layers and
    # stronger regularization to balance capacity and overfitting risk.
    'very_deep_mlp': {
        'model__hidden_layer_sizes': [
            (512, 256, 128, 64),
            (512, 256, 128, 64, 32)
        ],
        'model__alpha': [1e-4, 1e-3, 1e-2],
        'model__learning_rate_init': [1e-3, 5e-3]
    },
}


def _parse_blend_weights(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in s.split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                out[k.strip()] = float(v)
            except:
                pass
    t = sum(out.values()) or 1.0
    return {k: v / t for k, v in out.items()}


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


def train_with_cv(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    base_model: Pipeline,
    param_grid: dict[str, list] | None,
    sample_weight: np.ndarray | None = None,
    group: np.ndarray | None = None
) -> Pipeline:
    """
    Train a model using time-series cross-validation and return the best estimator.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.  Must contain the feature columns only (no ts/symbol).
    y_train : np.ndarray
        Target values corresponding to X_train.
    base_model : Pipeline
        Sklearn Pipeline to tune.  The estimator must be accessible via the 'model' step.
    param_grid : dict[str, list], optional
        Hyperparameter grid for GridSearchCV.  Keys should correspond to pipeline
        parameter names (e.g. 'model__n_estimators').  If None or empty, the
        base_model will be fit without hyper‑parameter search.
    sample_weight : np.ndarray, optional
        Sample weights for each observation.  If provided, these weights will be
        passed through to each fit.
    group : np.ndarray, optional
        Optional group array for learning‑to‑rank models (e.g. XGBoost rank:ndcg).

    Returns
    -------
    Pipeline
        The fitted best estimator.  If no param_grid is provided, returns the base_model after fitting.
    """
    if param_grid:
        cv = TimeSeriesSplit(n_splits=5)
        # Select search strategy based on USE_RANDOM_SEARCH flag.
        if USE_HALVING_SEARCH:
            # Successive halving search: adaptively allocates resources to promising hyper-parameter configurations.
            # Use the number of samples as the resource and stop early for poor performers.
            search = HalvingRandomSearchCV(
                base_model,
                param_distributions=param_grid,
                cv=cv,
                factor=HALVING_FACTOR,
                resource='n_samples',
                max_resources='auto',
                # Use 'smallest' for min_resources to avoid invalid 'auto' default
                min_resources='smallest',
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
        elif USE_RANDOM_SEARCH:
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_grid,
                n_iter=N_RANDOM_SEARCH_ITER,
                cv=cv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=0,
                random_state=42,
            )
        else:
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=0,
            )
        fit_params: dict[str, Any] = {}
        if sample_weight is not None:
            fit_params['model__sample_weight'] = sample_weight
        # Pass group information only when tuning an LTR model (objectives containing 'rank')
        if group is not None:
            fit_params['model__group'] = group
        try:
            search.fit(X_train, y_train, **fit_params)
            return search.best_estimator_
        except Exception as e:
            log.warning(f"Hyperparameter search failed ({e}); falling back to base model.")
            # Fall back to a simple fit with provided weights
            try:
                if fit_params:
                    base_model.fit(X_train, y_train, **fit_params)
                else:
                    base_model.fit(X_train, y_train)
            except Exception:
                base_model.fit(X_train, y_train)
            return base_model
    else:
        # No tuning; fit base model directly
        if sample_weight is not None:
            return base_model.fit(X_train, y_train, model__sample_weight=sample_weight)  # type: ignore[call-arg]
        else:
            return base_model.fit(X_train, y_train)


def _select_target_with_fallback(df: pd.DataFrame, primary_target: str, fallback_target: str, min_coverage: float) -> str:
    """
    Select the best target variable based on coverage thresholds.
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


def _ic_by_model(train_df: pd.DataFrame, feature_cols: list[str], target_variable: str = TARGET_VARIABLE) -> Dict[str, float]:
    """
    Compute information coefficients (IC) for each model over a recent evaluation window.
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
            fit_params: Dict[str, Any] = {}
            # For LTR models (identified by 'ltr' in name), pass group parameter
            if 'ltr' in name:
                fit_params['model__group'] = group_sizes
            # Fit on full training set (sorted by timestamp for LTR)
            if len(sorted_idx) == len(X_train):
                X_train_sorted = X_train.loc[sorted_idx, feature_cols]
                y_train_sorted = y_train.loc[sorted_idx].values
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
    return {k: v / s for k, v in pos.items()}


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
    # Merge sparse AltSignals (if any).  These event-driven or sentiment signals
    # are pivoted into wide columns.  If INCLUDE_ALT_SIGNALS is enabled via
    # environment variable, we will treat these columns as additional features.
    alts = _load_altsignals(start_ts, latest_ts)
    include_alt = os.getenv('INCLUDE_ALT_SIGNALS', 'false').lower() in ('1', 'true', 'yes')
    if not alts.empty:
        feats = feats.merge(alts, on=['symbol', 'ts'], how='left')
    # Determine dynamic feature columns starting from global FEATURE_COLS.
    feature_cols = list(FEATURE_COLS)
    if include_alt and not alts.empty:
        alt_cols = [c for c in alts.columns if c not in ('symbol', 'ts')]
        # Append raw alt signal columns
        feature_cols += [c for c in alt_cols if c not in feature_cols]
        # Compute lag features for each alt signal.  Lagging event-driven
        # signals captures the short-term reaction to news or events.  The
        # shift is computed per symbol to avoid forward-looking bias.
        for c in alt_cols:
            lag_col = f"{c}_lag1"
            if lag_col not in feats.columns:
                feats[lag_col] = feats.groupby('symbol')[c].shift(1)
            if lag_col not in feature_cols:
                feature_cols.append(lag_col)
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
    # Fill missing feature columns with zeros (for sparse features) and NaN with zeros.
    # Use the dynamic feature_cols list that may include alt signals.
    for c in feature_cols:
        if c not in train_df.columns:
            train_df[c] = 0.0
        if c not in latest_df.columns:
            latest_df[c] = 0.0
    train_df[feature_cols] = train_df[feature_cols].fillna(0.0)
    latest_df[feature_cols] = latest_df[feature_cols].fillna(0.0)
    # Prepare training matrix and target
    ID_COLS = ['ts', 'symbol']
    X = train_df[ID_COLS + feature_cols]
    y = train_df[selected_target].values
    X_latest = latest_df[feature_cols]
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
        X_train_sorted = X.loc[sorted_idx, feature_cols]
        y_train_sorted = train_df.loc[sorted_idx, selected_target].values
        sample_weights = sample_weights.loc[sorted_idx]
    else:
        sorted_idx = X.index
        X_train_sorted = X[feature_cols]
        y_train_sorted = y
    # Group sizes: number of samples per date in sorted training set
    group_sizes = X.loc[sorted_idx].groupby('ts').size().values
    models = _model_specs()
    outputs: Dict[str, Any] = {}
    preds_dict: Dict[str, pd.Series] = {}
    # Collect training predictions from each base model for meta blending.
    train_preds_dict: Dict[str, np.ndarray] = {}
    # Compute blend weights from config and adaptive IC weighting
    from config import BLEND_WEIGHTS, REGIME_GATING
    blend_w = _parse_blend_weights(BLEND_WEIGHTS)
    ic_w = _ic_by_model(train_df[['ts', 'symbol', selected_target] + feature_cols], feature_cols, selected_target)
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
    prediction_horizon = int(TARGET_HORIZON_DAYS)
    created_at_ts = as_naive_utc(pd.Timestamp.utcnow().floor('s'))
    # Prepare a timestamp for created_at
    created_at_ts = pd.Timestamp.now()
    # Fit & predict base models
    for name, pipe in models.items():
        # Deep copy the pipeline to avoid contaminating base specs
        base_model = copy.deepcopy(pipe)
        # Determine hyper‑parameter grid (if any)
        param_grid = PARAM_GRIDS.get(name)
        # Prepare fit parameters for sample weights and group sizes
        # Note: sample_weights and group_sizes are aligned to sorted_idx
        fit_params: Dict[str, Any] = {}
        if sample_weights is not None:
            fit_params['model__sample_weight'] = sample_weights.values
        if 'ltr' in name:
            fit_params['model__group'] = group_sizes
        try:
            if param_grid:
                # Use cross‑validated hyper‑parameter search to find the best pipeline
                tuned_model = train_with_cv(
                    X_train_sorted,
                    y_train_sorted,
                    base_model,
                    param_grid,
                    sample_weight=sample_weights.values if sample_weights is not None else None,
                    group=group_sizes if 'ltr' in name else None
                )
                p2 = tuned_model
            else:
                # Fit without tuning
                if fit_params:
                    p2 = base_model.fit(X_train_sorted, y_train_sorted, **fit_params)
                else:
                    p2 = base_model.fit(X_train_sorted, y_train_sorted)
        except TypeError as e:
            # Some models may not support sample_weight; retry without it
            if 'sample_weight' in str(e) and 'model__sample_weight' in fit_params:
                log.warning(f"Model {name} does not support sample_weight. Training without weights.")
                try:
                    if param_grid:
                        p2 = train_with_cv(
                            X_train_sorted,
                            y_train_sorted,
                            base_model,
                            param_grid,
                            sample_weight=None,
                            group=group_sizes if 'ltr' in name else None
                        )
                    else:
                        p2 = base_model.fit(X_train_sorted, y_train_sorted)
                except Exception as ex:
                    log.error(f"Error during fallback training for model {name}: {ex}")
                    continue
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
        # Predict on the training set for meta-model training
        try:
            train_pred_vals = p2.predict(X_train_sorted)
            train_preds_dict[name] = train_pred_vals.astype(float)
        except Exception as e:
            # If a model cannot produce training predictions (e.g. due to incompatible input shape), skip it.
            log.warning(f"Training prediction failed for model {name}: {e}; model will be excluded from meta blend.")
        out = latest_df[['symbol']].copy()
        out['ts'] = latest_ts
        out['y_pred'] = pred_vals.astype(float)
        out['model_version'] = f"{name}_v1"
        # Add horizon and created_at columns
        out['horizon'] = TARGET_HORIZON_DAYS
        out['created_at'] = created_at_ts
        preds_dict[name] = out.set_index('symbol')['y_pred']
        enriched_out = with_prediction_metadata(out, prediction_horizon, created_at_ts)
        outputs[name] = enriched_out.copy()
        # Persist predictions
        upsert_dataframe(
            enriched_out[['symbol', 'ts', 'y_pred', 'model_version', 'horizon', 'created_at']],
            Prediction,
            ['symbol', 'ts', 'model_version']
        )
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
        out = with_prediction_metadata(out, prediction_horizon, created_at_ts)
        outputs['blend_raw'] = out.copy()
        upsert_dataframe(out[['symbol', 'ts', 'y_pred', 'model_version', 'horizon', 'created_at']], Prediction, ['symbol', 'ts', 'model_version'])
        out['horizon'] = TARGET_HORIZON_DAYS
        out['created_at'] = created_at_ts
        outputs['blend_raw'] = out.copy()
        upsert_dataframe(
            out[['symbol', 'ts', 'y_pred', 'model_version', 'horizon', 'created_at']],
            Prediction,
            ['symbol', 'ts', 'model_version']
        )
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
            out2 = with_prediction_metadata(out2, prediction_horizon, created_at_ts)
            outputs['blend'] = out2.copy()
            upsert_dataframe(out2[['symbol', 'ts', 'y_pred', 'model_version', 'horizon', 'created_at']], Prediction, ['symbol', 'ts', 'model_version'])
            upsert_dataframe(
                out2[['symbol', 'ts', 'y_pred', 'model_version', 'horizon', 'created_at']],
                Prediction,
                ['symbol', 'ts', 'model_version']
            )
        except Exception as e:
            log.warning(f"Neutralization failed: {e}; fallback to raw blend.")
            out_fallback = out.copy()
            out_fallback['model_version'] = 'blend_v1'
            out_fallback = with_prediction_metadata(out_fallback, prediction_horizon, created_at_ts)
            upsert_dataframe(out_fallback[['symbol', 'ts', 'y_pred', 'model_version', 'horizon', 'created_at']], Prediction, ['symbol', 'ts', 'model_version'])

            upsert_dataframe(
                out_fallback[['symbol', 'ts', 'y_pred', 'model_version', 'horizon', 'created_at']],
                Prediction,
                ['symbol', 'ts', 'model_version']
            )
    # ----------------------------------------------------------------------
    # Meta‑model blending.  Combine base model predictions via a linear
    # meta‑model (Ridge regression) trained on the out-of-sample predictions
    # from each base model.  This approach automatically learns weights
    # rather than relying solely on heuristic blending.  Only perform
    # meta blending if we have collected predictions for at least two models.
    if train_preds_dict and len(train_preds_dict) >= 2:
        try:
            from sklearn.linear_model import RidgeCV
            # Build matrix of training predictions (rows correspond to y_train_sorted).  The order of columns is defined by keys.
            meta_keys = list(train_preds_dict.keys())
            meta_X_train = np.column_stack([train_preds_dict[k] for k in meta_keys])
            # Time-series cross-validation for meta model
            tscv_meta = TimeSeriesSplit(n_splits=5)
            meta_reg = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=tscv_meta, scoring='neg_mean_absolute_error')
            meta_reg.fit(meta_X_train, y_train_sorted)
            # Log learned coefficients for transparency
            coeffs = {k: float(c) for k, c in zip(meta_keys, meta_reg.coef_)}
            log.info(f"Meta blend coefficients (alpha={meta_reg.alpha_}): {coeffs}")
            # Prepare matrix of latest predictions using the same keys
            meta_X_latest = np.column_stack([
                preds_dict[k].reindex(latest_df['symbol']).fillna(0.0).values
                for k in meta_keys
            ])
            meta_pred = meta_reg.predict(meta_X_latest)
            meta_out = latest_df[['symbol']].copy()
            meta_out['ts'] = latest_ts
            meta_out['y_pred'] = meta_pred.astype(float)
            meta_out['model_version'] = 'blend_meta_v1'
            meta_out = with_prediction_metadata(meta_out, prediction_horizon, created_at_ts)
            outputs['blend_meta'] = meta_out.copy()
            upsert_dataframe(meta_out[['symbol', 'ts', 'y_pred', 'model_version', 'horizon', 'created_at']], Prediction, ['symbol', 'ts', 'model_version'])
            meta_out['horizon'] = TARGET_HORIZON_DAYS
            meta_out['created_at'] = created_at_ts
            outputs['blend_meta'] = meta_out.copy()
            upsert_dataframe(
                meta_out[['symbol', 'ts', 'y_pred', 'model_version', 'horizon', 'created_at']],
                Prediction,
                ['symbol', 'ts', 'model_version']
            )
        except Exception as e:
            log.warning(f"Meta blending failed: {e}")
    # ------------------------------------------------------------------
    # Regime‑aware gating: blend tree and neural model predictions based
    # on the current volatility regime.  This is applied only for the
    # latest prediction date and requires that at least one tree and one
    # neural model have produced predictions.  The gating factor is a
    # logistic function of the cross‑sectional mean 21‑day volatility
    # compared to a threshold.  A higher volatility regime results in
    # greater weight on tree‑based models, while a lower regime weights
    # neural models more heavily.
    if USE_REGIME_GATING and preds_dict and 'vol_21' in latest_df.columns:
        try:
            # Current cross‑sectional volatility
            current_vol = float(latest_df['vol_21'].mean())
            if not np.isnan(current_vol):
                # Compute gating factor in (0,1)
                denom = GATING_THRESHOLD if GATING_THRESHOLD != 0 else 1.0
                x = (current_vol - GATING_THRESHOLD) / denom
                gating_factor = 1.0 / (1.0 + np.exp(-GATING_SLOPE * x))
                # Separate predictions into tree and neural groups
                tree_keys = {k for k in preds_dict.keys() if k in {
                    'rf', 'extra_trees', 'gbr', 'xgb_reg', 'lgbm', 'cat', 'hgb'
                }}
                neural_keys = {k for k in preds_dict.keys() if k in {
                    'ridge', 'elasticnet', 'mlp', 'deep_mlp', 'q_table', 'very_deep_mlp'
                }}
                # Compute average predictions for each group
                tree_pred = None
                if tree_keys:
                    arrs = []
                    for k in tree_keys:
                        arrs.append(preds_dict[k].reindex(latest_df['symbol']).fillna(0.0).values)
                    tree_pred = np.mean(arrs, axis=0)
                neural_pred = None
                if neural_keys:
                    arrs = []
                    for k in neural_keys:
                        arrs.append(preds_dict[k].reindex(latest_df['symbol']).fillna(0.0).values)
                    neural_pred = np.mean(arrs, axis=0)
                if tree_pred is not None and neural_pred is not None:
                    gating_pred = gating_factor * tree_pred + (1.0 - gating_factor) * neural_pred
                    regime_out = latest_df[['symbol']].copy()
                    regime_out['ts'] = latest_ts
                    regime_out['y_pred'] = gating_pred.astype(float)
                    regime_out['model_version'] = 'blend_regime_v1'
                    regime_out = with_prediction_metadata(regime_out, prediction_horizon, created_at_ts)
                    outputs['blend_regime'] = regime_out.copy()
                    upsert_dataframe(regime_out[['symbol', 'ts', 'y_pred', 'model_version', 'horizon', 'created_at']], Prediction, ['symbol', 'ts', 'model_version'])
                    regime_out['horizon'] = TARGET_HORIZON_DAYS
                    regime_out['created_at'] = created_at_ts
                    outputs['blend_regime'] = regime_out.copy()
                    upsert_dataframe(
                        regime_out[['symbol', 'ts', 'y_pred', 'model_version', 'horizon', 'created_at']],
                        Prediction,
                        ['symbol', 'ts', 'model_version']
                    )
                else:
                    log.warning("Regime gating skipped: insufficient tree or neural predictions")
        except Exception as e:
            log.warning(f"Regime gating failed: {e}")
    log.info("Live training and prediction complete.")
    return outputs

from config import PREFERRED_MODEL
from db import BacktestEquity
from performance.metrics import compute_all_metrics

def run_walkforward_backtest(model_version: str | None = None, top_n: int = 20) -> pd.DataFrame:
    """
    Simple walkforward backtest: for each ts, take top-N positive names by y_pred (given model_version),
    realize fwd_ret over TARGET_HORIZON_DAYS, average cross-sectionally, and
    cumulate to equity.  Persists results to backtest_equity.
    """
    mv = model_version or PREFERRED_MODEL
    with engine.connect() as con:
        preds = pd.read_sql_query(
            text("SELECT symbol, ts, y_pred FROM predictions WHERE model_version = :mv ORDER BY ts, y_pred DESC"),
            con,
            params={"mv": mv},
            parse_dates=['ts']
        )
    if preds.empty:
        log.warning("No predictions found for backtest.")
        return pd.DataFrame(columns=['ts', 'equity', 'daily_return', 'drawdown', 'tcost_impact'])
    # Compute realized forward returns
    with engine.connect() as con:
        px = pd.read_sql_query(
            text(f"SELECT symbol, ts, {price_expr()} AS px FROM daily_bars"),
            con,
            parse_dates=['ts']
        )
    if px.empty:
        log.warning("No prices for backtest.")
        return pd.DataFrame(columns=['ts', 'equity', 'daily_return', 'drawdown', 'tcost_impact'])
    px = px.sort_values(['symbol', 'ts'])
    px['px_fwd'] = px.groupby('symbol')['px'].shift(-TARGET_HORIZON_DAYS)
    px['fwd_ret'] = (px['px_fwd'] / px['px']) - 1.0
    df = preds.merge(px[['symbol', 'ts', 'fwd_ret']], on=['symbol', 'ts'], how='left')
    if df['fwd_ret'].isna().all():
        log.warning("No forward returns matched for backtest.")
        return pd.DataFrame(columns=['ts', 'equity', 'daily_return', 'drawdown', 'tcost_impact'])
    # Pick top-N positive per ts
    port = (
        df[df['y_pred'] > 0]
        .sort_values(['ts', 'y_pred'], ascending=[True, False])
        .groupby('ts')
        .head(top_n)
        .groupby('ts')['fwd_ret'].mean()
        .dropna()
        .sort_index()
    )
    if port.empty:
        return pd.DataFrame(columns=['ts', 'equity', 'daily_return', 'drawdown', 'tcost_impact'])
    equity = (1.0 + port).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    out = pd.DataFrame({
        'ts': port.index.normalize(),
        'equity': equity.values,
        'daily_return': port.values,
        'drawdown': drawdown.values,
        'tcost_impact': 0.0
    })
    # Persist
    upsert_dataframe(out, BacktestEquity, ['ts'])
    # Compute and log performance metrics
    metrics = compute_all_metrics(port)
    log.info(f"Backtest metrics: {metrics}")
    return out

