"""
Enhanced Trade Generation with Abort Mechanisms

Provides robust trade generation with comprehensive error handling,
abort mechanisms, and fail-safe operations.
"""
from __future__ import annotations
import logging
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import date, datetime
from market_calendar import should_run_pipeline, is_market_day
from warning_dedup import warn_once

log = logging.getLogger(__name__)

def _to_date(x) -> date:
    """
    Helper function to safely convert various datetime types to date.
    
    Args:
        x: Input that could be datetime.datetime, datetime.date, pandas.Timestamp, or string
        
    Returns:
        datetime.date object
    """
    if isinstance(x, date) and not isinstance(x, datetime):
        # Already a date object (not datetime)
        return x
    elif isinstance(x, datetime):
        # datetime.datetime -> date
        return x.date()
    elif hasattr(x, 'to_pydatetime'):
        # pandas.Timestamp -> date
        return x.to_pydatetime().date()
    elif isinstance(x, str):
        # String -> date
        try:
            return datetime.fromisoformat(x).date()
        except ValueError:
            return datetime.strptime(x, "%Y-%m-%d").date()
    else:
        # Fallback: try to convert to string first
        return datetime.fromisoformat(str(x)).date()

class TradeGenerationError(Exception):
    """Custom exception for trade generation failures."""
    pass

class TradeGenerationAbort(Exception):
    """Exception for deliberate trade generation aborts."""
    pass

def validate_trading_conditions() -> Tuple[bool, str]:
    """
    Validate that conditions are suitable for trade generation.
    
    Returns:
        Tuple of (is_valid: bool, reason: str)
    """
    try:
        # Check if it's a valid trading day
        should_run, reason = should_run_pipeline()
        if not should_run:
            return False, f"Not a valid trading day: {reason}"
        
        # Check if we have recent market data
        from sqlalchemy import text, create_engine
        from config import DATABASE_URL
        
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            # Check latest market data date
            result = conn.execute(text("SELECT MAX(ts) FROM daily_bars")).fetchone()
            if result[0] is None:
                return False, "No market data available in database"
            
            latest_data_date = _to_date(result[0])
            
            # Check if data is reasonably recent (within 5 trading days)
            days_since_data = (date.today() - latest_data_date).days
            if days_since_data > 7:  # More than a week old
                return False, f"Market data is stale (last update: {latest_data_date})"
        
        return True, "Trading conditions are valid"
        
    except Exception as e:
        return False, f"Validation failed: {str(e)}"

def validate_predictions_quality(predictions_df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that predictions are of sufficient quality for trading.
    
    Args:
        predictions_df: DataFrame with predictions
        
    Returns:
        Tuple of (is_valid: bool, reason: str)
    """
    if predictions_df.empty:
        return False, "No predictions available"
    
    # Check for reasonable prediction spread
    pred_std = predictions_df['y_pred'].std()
    if pred_std < 0.001:  # Very low variance - might indicate a problem
        warn_once(log, "low_prediction_variance", 
                 f"Low prediction variance detected: {pred_std:.6f}", "critical")
    
    # Check for extreme predictions that might indicate model issues
    extreme_threshold = 0.5  # 50% return prediction is suspicious
    extreme_preds = predictions_df[abs(predictions_df['y_pred']) > extreme_threshold]
    
    if len(extreme_preds) > len(predictions_df) * 0.1:  # More than 10% extreme
        return False, f"Too many extreme predictions (>{extreme_threshold*100}%): {len(extreme_preds)}"
    
    # Check prediction freshness
    unique_dates = predictions_df['ts'].unique()
    if len(unique_dates) > 1:
        warn_once(log, "mixed_prediction_dates",
                 f"Predictions from multiple dates: {unique_dates}", "default")
    
    return True, f"Predictions quality acceptable ({len(predictions_df)} symbols)"

def validate_portfolio_construction(weights: pd.Series, predictions_df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that portfolio construction results are reasonable.
    
    Args:
        weights: Portfolio weights
        predictions_df: Original predictions
        
    Returns:
        Tuple of (is_valid: bool, reason: str)
    """
    if weights.empty:
        return False, "No portfolio weights generated"
    
    # Check weight constraints
    total_weight = abs(weights).sum()
    if total_weight > 2.0:  # More than 200% gross exposure
        return False, f"Excessive gross exposure: {total_weight:.2f}"
    
    # Check for concentration
    max_weight = abs(weights).max()
    if max_weight > 0.15:  # More than 15% in single position
        return False, f"Excessive concentration: {max_weight:.2f} in single position"
    
    # Check that weights align with predictions
    if len(weights) > len(predictions_df) * 2:  # More than 2x positions vs predictions
        warn_once(log, "weight_prediction_mismatch",
                 f"Many more weights ({len(weights)}) than predictions ({len(predictions_df)})", "default")
    
    return True, f"Portfolio construction valid ({len(weights)} positions, {total_weight:.2f} gross)"

def enhanced_generate_today_trades() -> pd.DataFrame:
    """
    Enhanced trade generation with comprehensive validation and abort mechanisms.
    
    Returns:
        DataFrame with generated trades or empty DataFrame if aborted
    """
    log.info("🚀 Starting enhanced trade generation with robustness checks")
    
    try:
        # Phase 1: Pre-flight validation
        conditions_valid, conditions_reason = validate_trading_conditions()
        if not conditions_valid:
            log.warning(f"⚠️ Trade generation aborted: {conditions_reason}")
            raise TradeGenerationAbort(conditions_reason)
        
        log.info(f"✅ Trading conditions validated: {conditions_reason}")
        
        # Phase 2: Load and validate predictions
        from trading.generate_trades import _load_latest_predictions
        
        try:
            predictions_df = _load_latest_predictions()
        except Exception as e:
            log.error(f"❌ Failed to load predictions: {e}")
            raise TradeGenerationError(f"Prediction loading failed: {e}")
        
        pred_valid, pred_reason = validate_predictions_quality(predictions_df)
        if not pred_valid:
            log.error(f"❌ Prediction quality check failed: {pred_reason}")
            raise TradeGenerationAbort(pred_reason)
        
        log.info(f"✅ Predictions validated: {pred_reason}")
        
        # Phase 3: Portfolio construction with validation
        from trading.generate_trades import _load_tca_cols, _get_current_shares, _get_system_nav
        from portfolio.optimizer import build_portfolio
        
        try:
            # Load supplementary data
            sup = _load_tca_cols(predictions_df["symbol"].tolist())
            pred_df = predictions_df.merge(sup, on=["symbol"], how="left")
            
            # Get current state
            today = date.today()
            current_shares = _get_current_shares()
            current_nav = _get_system_nav()
            
            if current_nav <= 0:
                raise TradeGenerationError(f"Invalid NAV: {current_nav}")
            
            # Build portfolio
            weights = build_portfolio(
                pred_df, today, current_symbols=current_shares.index.tolist()
            )
            
        except Exception as e:
            log.error(f"❌ Portfolio construction failed: {e}")
            raise TradeGenerationError(f"Portfolio construction failed: {e}")
        
        portfolio_valid, portfolio_reason = validate_portfolio_construction(weights, predictions_df)
        if not portfolio_valid:
            log.error(f"❌ Portfolio validation failed: {portfolio_reason}")
            raise TradeGenerationAbort(portfolio_reason)
        
        log.info(f"✅ Portfolio construction validated: {portfolio_reason}")
        
        # Phase 4: Trade generation with price validation
        try:
            from trading.generate_trades import generate_today_trades as original_generate_trades
            trades_df = original_generate_trades()
            
            # Validate generated trades
            if not trades_df.empty:
                # Check for reasonable quantities
                large_trades = trades_df[trades_df['quantity'] > current_nav * 0.1]  # More than 10% of NAV
                if not large_trades.empty:
                    warn_once(log, "large_trade_warning",
                             f"Large trades detected: {len(large_trades)} trades > 10% NAV", "critical")
                
                # Check for price sanity
                invalid_prices = trades_df[~trades_df['price'].between(0.01, 10000)]  # Basic sanity check
                if not invalid_prices.empty:
                    log.error(f"❌ Invalid prices detected in {len(invalid_prices)} trades")
                    raise TradeGenerationError(f"Invalid prices in trades: {invalid_prices['price'].tolist()}")
            
            log.info(f"✅ Generated {len(trades_df)} validated trades")
            return trades_df
            
        except Exception as e:
            log.error(f"❌ Trade generation execution failed: {e}")
            raise TradeGenerationError(f"Trade execution failed: {e}")
    
    except TradeGenerationAbort as e:
        log.warning(f"🛑 Trade generation deliberately aborted: {e}")
        return pd.DataFrame(columns=[
            "id", "symbol", "side", "quantity", "price", "status", "trade_date"
        ])
    
    except TradeGenerationError as e:
        log.error(f"💥 Trade generation failed with error: {e}")
        return pd.DataFrame(columns=[
            "id", "symbol", "side", "quantity", "price", "status", "trade_date"
        ])
    
    except Exception as e:
        log.exception(f"💥 Unexpected error in trade generation: {e}")
        return pd.DataFrame(columns=[
            "id", "symbol", "side", "quantity", "price", "status", "trade_date"
        ])

def get_trade_generation_health() -> Dict[str, Any]:
    """
    Get comprehensive health check for trade generation system.
    
    Returns:
        Dictionary with health status information
    """
    health = {
        'timestamp': date.today().isoformat(),
        'trading_conditions': {},
        'data_quality': {},
        'system_state': {},
        'overall_status': 'unknown'
    }
    
    try:
        # Trading conditions
        conditions_valid, conditions_reason = validate_trading_conditions()
        health['trading_conditions'] = {
            'valid': conditions_valid,
            'reason': conditions_reason
        }
        
        # Data quality checks
        try:
            from trading.generate_trades import _load_latest_predictions
            predictions_df = _load_latest_predictions()
            pred_valid, pred_reason = validate_predictions_quality(predictions_df)
            health['data_quality'] = {
                'predictions_valid': pred_valid,
                'predictions_reason': pred_reason,
                'prediction_count': len(predictions_df)
            }
        except Exception as e:
            health['data_quality'] = {
                'predictions_valid': False,
                'predictions_reason': f"Failed to load: {e}",
                'prediction_count': 0
            }
        
        # System state
        try:
            from trading.generate_trades import _get_current_shares, _get_system_nav
            current_shares = _get_current_shares()
            current_nav = _get_system_nav()
            health['system_state'] = {
                'nav': current_nav,
                'position_count': len(current_shares),
                'nav_valid': current_nav > 0
            }
        except Exception as e:
            health['system_state'] = {
                'nav': 0,
                'position_count': 0,
                'nav_valid': False,
                'error': str(e)
            }
        
        # Overall status
        if (health['trading_conditions']['valid'] and 
            health['data_quality']['predictions_valid'] and 
            health['system_state']['nav_valid']):
            health['overall_status'] = 'healthy'
        elif health['trading_conditions']['valid']:
            health['overall_status'] = 'degraded'
        else:
            health['overall_status'] = 'unhealthy'
    
    except Exception as e:
        health['overall_status'] = 'error'
        health['error'] = str(e)
    
    return health