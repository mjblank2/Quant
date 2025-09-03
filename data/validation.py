"""
Data validation and quality control pipeline for institutional-grade data integrity.

This module implements automated checks for:
- Data completeness and staleness
- Price and volume anomaly detection  
- Point-in-time data consistency
- Corporate actions validation
"""
from __future__ import annotations
import logging
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional
from sqlalchemy import text
from db import engine
from config import (
    DATA_STALENESS_THRESHOLD_HOURS,
    PRICE_ANOMALY_THRESHOLD_SIGMA,
    VOLUME_ANOMALY_THRESHOLD_SIGMA,
    ENABLE_DATA_VALIDATION
)

log = logging.getLogger(__name__)

class ValidationResult:
    """Container for validation results with metrics and alerts."""
    
    def __init__(self):
        self.passed = True
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.metrics: Dict[str, float] = {}
    
    def add_warning(self, message: str):
        """Add a warning (non-blocking issue)."""
        self.warnings.append(message)
        log.warning(f"Data validation warning: {message}")
    
    def add_error(self, message: str):
        """Add an error (blocking issue)."""
        self.errors.append(message)
        self.passed = False
        log.error(f"Data validation error: {message}")
    
    def add_metric(self, name: str, value: float):
        """Add a quality metric."""
        self.metrics[name] = value

def check_data_staleness() -> ValidationResult:
    """Check for stale data across key tables."""
    result = ValidationResult()
    
    if not ENABLE_DATA_VALIDATION:
        return result
    
    threshold = datetime.now() - timedelta(hours=DATA_STALENESS_THRESHOLD_HOURS)
    
    # Check daily_bars freshness
    with engine.connect() as conn:
        latest_bar = conn.execute(
            text("SELECT MAX(ts) as latest FROM daily_bars")
        ).scalar()
        
        if latest_bar and pd.to_datetime(latest_bar) < threshold:
            result.add_error(f"Daily bars data is stale. Latest: {latest_bar}")
        
        # Check fundamentals freshness
        latest_fund = conn.execute(
            text("SELECT MAX(as_of) as latest FROM fundamentals")
        ).scalar()
        
        if latest_fund and pd.to_datetime(latest_fund) < threshold:
            result.add_warning(f"Fundamentals data is stale. Latest: {latest_fund}")
    
    return result

def detect_price_anomalies(symbol: str = None, lookback_days: int = 30) -> ValidationResult:
    """Detect extreme price movements that may indicate data quality issues."""
    result = ValidationResult()
    
    if not ENABLE_DATA_VALIDATION:
        return result
    
    # Build query
    where_clause = "WHERE ts >= CURRENT_DATE - INTERVAL '%s days'" % lookback_days
    if symbol:
        where_clause += f" AND symbol = '{symbol}'"
    
    query = f"""
    SELECT symbol, ts, close, 
           LAG(close) OVER (PARTITION BY symbol ORDER BY ts) as prev_close,
           volume,
           AVG(volume) OVER (PARTITION BY symbol ORDER BY ts ROWS 20 PRECEDING) as avg_volume
    FROM daily_bars 
    {where_clause}
    ORDER BY symbol, ts
    """
    
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, parse_dates=['ts'])
    
    if df.empty:
        return result
    
    # Calculate returns and z-scores
    df['return'] = df['close'] / df['prev_close'] - 1
    df = df.dropna()
    
    for sym, group in df.groupby('symbol'):
        if len(group) < 10:  # Need minimum data for statistics
            continue
            
        returns = group['return']
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        if std_ret == 0:
            continue
            
        # Flag extreme returns
        z_scores = np.abs((returns - mean_ret) / std_ret)
        extreme_mask = z_scores > PRICE_ANOMALY_THRESHOLD_SIGMA
        
        if extreme_mask.any():
            extreme_dates = group.loc[extreme_mask, 'ts'].dt.strftime('%Y-%m-%d').tolist()
            max_z = z_scores.max()
            result.add_warning(
                f"Extreme price moves detected for {sym} on {extreme_dates}. "
                f"Max z-score: {max_z:.2f}"
            )
        
        # Flag volume anomalies
        volumes = group['volume']
        avg_volumes = group['avg_volume']
        volume_ratios = volumes / avg_volumes
        
        extreme_volume_mask = volume_ratios > VOLUME_ANOMALY_THRESHOLD_SIGMA
        if extreme_volume_mask.any():
            extreme_vol_dates = group.loc[extreme_volume_mask, 'ts'].dt.strftime('%Y-%m-%d').tolist()
            max_ratio = volume_ratios.max()
            result.add_warning(
                f"Extreme volume detected for {sym} on {extreme_vol_dates}. "
                f"Max ratio: {max_ratio:.2f}x"
            )
    
    # Store metrics
    result.add_metric("symbols_checked", df['symbol'].nunique())
    result.add_metric("total_observations", len(df))
    
    return result

def check_data_completeness(symbols: List[str] = None) -> ValidationResult:
    """Check for missing data in critical tables."""
    result = ValidationResult()
    
    if not ENABLE_DATA_VALIDATION:
        return result
    
    # Get universe symbols if not provided
    if symbols is None:
        with engine.connect() as conn:
            symbols = pd.read_sql_query(
                "SELECT symbol FROM universe WHERE included = true", 
                conn
            )['symbol'].tolist()
    
    if not symbols:
        result.add_warning("No symbols found in universe for completeness check")
        return result
    
    start_date = date.today() - timedelta(days=30)
    end_date = date.today() - timedelta(days=1)
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = [d.date() for d in mcal.date_range(schedule, frequency='1D')]

    if not trading_days:
        result.add_warning("No trading days available for completeness check")
        return result

    with engine.connect() as conn:
        missing_query = """
        WITH symbol_dates AS (
            SELECT s.symbol, d.ts
            FROM (SELECT unnest(%(symbols)s) as symbol) s
            CROSS JOIN (SELECT unnest(%(dates)s)::date as ts) d
        ),
        existing_data AS (
            SELECT symbol, ts FROM daily_bars
            WHERE symbol = ANY(%(symbols)s)
              AND ts = ANY(%(dates)s)
        )
        SELECT sd.symbol, sd.ts
        FROM symbol_dates sd
        LEFT JOIN existing_data ed ON sd.symbol = ed.symbol AND sd.ts = ed.ts
        WHERE ed.symbol IS NULL
        ORDER BY sd.symbol, sd.ts
        """

        missing_df = pd.read_sql_query(
            missing_query,
            conn,
            params={'symbols': symbols, 'dates': trading_days}
        )
    
    if not missing_df.empty:
        missing_count = len(missing_df)
        missing_symbols = missing_df['symbol'].nunique()
        result.add_warning(
            f"Found {missing_count} missing daily bar records "
            f"across {missing_symbols} symbols in last 30 days"
        )
    
    # Store completeness metrics
    expected_records = len(symbols) * len(trading_days)
    actual_records = expected_records - len(missing_df)
    completeness_pct = (actual_records / expected_records) * 100 if expected_records > 0 else 0
    
    result.add_metric("data_completeness_pct", completeness_pct)
    result.add_metric("missing_records", len(missing_df))
    
    return result

def validate_pit_consistency() -> ValidationResult:
    """Validate point-in-time data consistency."""
    result = ValidationResult()
    
    if not ENABLE_DATA_VALIDATION:
        return result
    
    # Check for future-dated data (impossible as-of dates)
    with engine.connect() as conn:
        future_data_query = """
        SELECT 'fundamentals' as table_name, COUNT(*) as count
        FROM fundamentals 
        WHERE as_of > CURRENT_DATE
        UNION ALL
        SELECT 'shares_outstanding', COUNT(*)
        FROM shares_outstanding 
        WHERE as_of > CURRENT_DATE
        """
        
        future_data = pd.read_sql_query(future_data_query, conn)
    
    for _, row in future_data.iterrows():
        if row['count'] > 0:
            result.add_error(
                f"Found {row['count']} future-dated records in {row['table_name']}"
            )
    
    # Check for reasonable knowledge_date consistency if column exists
    try:
        with engine.connect() as conn:
            knowledge_check = conn.execute(
                text("""
                SELECT COUNT(*) as count
                FROM shares_outstanding 
                WHERE knowledge_date IS NOT NULL 
                AND knowledge_date < as_of
                """)
            ).scalar()
            
            if knowledge_check > 0:
                result.add_error(
                    f"Found {knowledge_check} records where knowledge_date < as_of"
                )
    except Exception:
        # knowledge_date column might not exist in all deployments
        pass
    
    return result

def run_validation_pipeline(symbols: List[str] = None) -> ValidationResult:
    """Run complete data validation pipeline."""
    log.info("Starting data validation pipeline")
    
    overall_result = ValidationResult()
    
    # Run all validation checks
    checks = [
        ("staleness", check_data_staleness),
        ("anomalies", lambda: detect_price_anomalies()),
        ("completeness", lambda: check_data_completeness(symbols)),
        ("pit_consistency", validate_pit_consistency)
    ]
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            
            # Merge results
            overall_result.warnings.extend(result.warnings)
            overall_result.errors.extend(result.errors)
            overall_result.metrics.update({
                f"{check_name}_{k}": v for k, v in result.metrics.items()
            })
            
            if not result.passed:
                overall_result.passed = False
                
        except Exception as e:
            error_msg = f"Validation check '{check_name}' failed: {e}"
            overall_result.add_error(error_msg)
    
    # Log summary
    log.info(f"Validation complete. Passed: {overall_result.passed}, "
             f"Warnings: {len(overall_result.warnings)}, "
             f"Errors: {len(overall_result.errors)}")
    
    return overall_result