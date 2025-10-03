"""
Enhanced data ingestion pipeline with institutional-grade data quality controls.

This module extends the existing ingestion pipeline with:
- TimescaleDB optimization setup
- Real-time data validation
- Data lineage tracking
- Point-in-time data integrity
"""
from __future__ import annotations
import logging
from datetime import datetime, date
from typing import List
import pandas as pd
from sqlalchemy import text

try:
    from db import engine, upsert_dataframe, DataValidationLog, DataLineage
except Exception:  # pragma: no cover - fallback for missing DB
    engine = None

    def upsert_dataframe(*args, **kwargs):
        return False

    class DataValidationLog:  # minimal placeholders
        __table__ = None

    class DataLineage:
        __table__ = None

from data.timescale import setup_timescaledb, get_timescaledb_info
from data.validation import run_validation_pipeline, ValidationResult
from config import ENABLE_DATA_VALIDATION, DEFAULT_KNOWLEDGE_LATENCY_DAYS

log = logging.getLogger(__name__)


def log_validation_result(result: ValidationResult, validation_type: str, symbols: List[str] = None):
    """Log validation results to database for auditing."""
    if not ENABLE_DATA_VALIDATION:
        return

    try:
        # Determine overall status
        if result.errors:
            status = "FAILED"
        elif result.warnings:
            status = "WARNING"
        else:
            status = "PASSED"

        # Create message summary
        message_parts = []
        if result.errors:
            message_parts.append(f"Errors: {len(result.errors)}")
        if result.warnings:
            message_parts.append(f"Warnings: {len(result.warnings)}")

        message = "; ".join(message_parts) if message_parts else "All checks passed"

        # Insert validation log
        validation_log = pd.DataFrame([{
            'run_timestamp': datetime.now(),
            'validation_type': validation_type,
            'status': status,
            'message': message,
            'metrics': result.metrics,
            'affected_symbols': symbols
        }])

        upsert_dataframe(
            validation_log,
            DataValidationLog.__table__,
            ['id'],  # Let autoincrement handle conflicts
        )

        log.info(f"Logged validation result: {validation_type} - {status}")

    except Exception as e:
        log.error(f"Failed to log validation result: {e}")


def log_data_lineage(table_name: str, symbols: List[str], data_date: date,
                     source: str, quality_score: float = None, lineage_metadata: dict = None):
    """Log data lineage for governance and auditing."""
    try:
        lineage_records = []

        for symbol in symbols:
            lineage_records.append({
                'table_name': table_name,
                'symbol': symbol,
                'data_date': data_date,
                'ingestion_timestamp': datetime.now(),
                'source': source,
                'source_timestamp': lineage_metadata.get('source_timestamp') if lineage_metadata else None,
                'quality_score': quality_score,
                'lineage_metadata': lineage_metadata
            })

        if lineage_records:
            lineage_df = pd.DataFrame(lineage_records)
            upsert_dataframe(
                lineage_df,
                DataLineage.__table__,
                ['table_name', 'symbol', 'data_date', 'source']
            )

            log.info(f"Logged lineage for {len(lineage_records)} records in {table_name}")

    except Exception as e:
        log.error(f"Failed to log data lineage: {e}")


def setup_infrastructure():
    """Set up TimescaleDB and other infrastructure optimizations."""
    log.info("Setting up data infrastructure")

    # Set up TimescaleDB
    timescale_success = setup_timescaledb()

    # Log infrastructure status
    timescale_info = get_timescaledb_info()
    log.info(f"TimescaleDB info: {timescale_info}")

    return timescale_success


def validate_and_ingest_daily_bars(df: pd.DataFrame, source: str = "unknown") -> bool:
    """
    Ingest daily bars data with validation and lineage tracking.

    Args:
        df: DataFrame with daily bars data
        source: Source identifier for lineage tracking

    Returns:
        True if ingestion successful, False otherwise
    """
    if df.empty:
        log.warning("Empty daily bars DataFrame provided")
        return True

    log.info(f"Ingesting {len(df)} daily bars records from {source}")

    # Pre-ingestion validation
    if ENABLE_DATA_VALIDATION:
        symbols = df['symbol'].unique().tolist()

        # Run anomaly detection on new data
        validation_result = run_validation_pipeline(symbols)
        log_validation_result(validation_result, "pre_ingestion", symbols)

        # Block ingestion if critical errors found
        if validation_result.errors:
            log.error("Critical validation errors found, blocking ingestion")
            return False

    try:
        # Perform the actual ingestion
        from data.ingest import DailyBar  # Import from existing module
        upsert_dataframe(df, DailyBar.__table__, ['symbol', 'ts'])

        # Log successful ingestion lineage
        if not df.empty:
            unique_symbols = df['symbol'].unique().tolist()
            data_dates = df['ts'].unique()

            for data_date in data_dates:
                log_data_lineage(
                    table_name='daily_bars',
                    symbols=unique_symbols,
                    data_date=pd.to_datetime(data_date).date(),
                    source=source,
                    quality_score=0.95,  # Default good quality score
                    lineage_metadata={'record_count': len(df[df['ts'] == data_date])}
                )

        # Post-ingestion validation
        if ENABLE_DATA_VALIDATION:
            post_validation = run_validation_pipeline(symbols)
            log_validation_result(post_validation, "post_ingestion", symbols)

        log.info(f"Successfully ingested daily bars from {source}")
        return True

    except Exception as e:
        log.error(f"Failed to ingest daily bars: {e}")
        return False


def validate_and_ingest_fundamentals(df: pd.DataFrame, source: str = "polygon") -> bool:
    """
    Ingest fundamentals data with bi-temporal tracking and validation.

    Args:
        df: DataFrame with fundamentals data
        source: Source identifier (e.g., 'polygon', 'bloomberg')

    Returns:
        True if ingestion successful, False otherwise
    """
    if df.empty:
        log.warning("Empty fundamentals DataFrame provided")
        return True

    log.info(f"Ingesting {len(df)} fundamentals records from {source}")

    # Add knowledge_date if not present (bi-temporal support)
    if 'knowledge_date' not in df.columns:
        if 'available_at' in df.columns:
            df['knowledge_date'] = df['available_at']
        else:
            # Default: knowledge available next day after as_of date
            df['knowledge_date'] = pd.to_datetime(df['as_of']) + pd.Timedelta(days=DEFAULT_KNOWLEDGE_LATENCY_DAYS)

    try:
        # Perform ingestion using proper ON CONFLICT upsert
        from data.fundamentals import _upsert_fundamentals
        _upsert_fundamentals(df)

        # Log lineage
        if not df.empty:
            unique_symbols = df['symbol'].unique().tolist()
            data_dates = df['as_of'].unique()

            for data_date in data_dates:
                log_data_lineage(
                    table_name='fundamentals',
                    symbols=unique_symbols,
                    data_date=pd.to_datetime(data_date).date(),
                    source=source,
                    quality_score=0.90,  # Fundamentals typically have good quality
                    lineage_metadata={
                        'record_count': len(df[df['as_of'] == data_date]),
                        'has_knowledge_date': 'knowledge_date' in df.columns
                    }
                )

        log.info(f"Successfully ingested fundamentals from {source}")
        return True

    except Exception as e:
        log.error(f"Failed to ingest fundamentals: {e}")
        return False


def run_infrastructure_health_check() -> dict:
    """Run comprehensive health check on data infrastructure."""
    health_status = {
        'overall_status': 'HEALTHY',
        'checks': {},
        'recommendations': []
    }

    try:
        # Check TimescaleDB status
        timescale_info = get_timescaledb_info()
        health_status['checks']['timescaledb'] = timescale_info

        if not timescale_info['available'] and timescale_info['enabled']:
            health_status['recommendations'].append(
                "TimescaleDB is enabled but not available. Consider installing TimescaleDB extension."
            )

        # Run data validation
        if ENABLE_DATA_VALIDATION:
            validation_result = run_validation_pipeline()
            health_status['checks']['data_validation'] = {
                'passed': validation_result.passed,
                'warnings': len(validation_result.warnings),
                'errors': len(validation_result.errors),
                'metrics': validation_result.metrics
            }

            if validation_result.errors:
                health_status['overall_status'] = 'CRITICAL'
                health_status['recommendations'].append(
                    f"Critical data validation errors detected: {len(validation_result.errors)} errors"
                )
            elif validation_result.warnings:
                health_status['overall_status'] = 'WARNING'

        # Check recent data freshness
        with engine.connect() as conn:
            latest_data = conn.execute(
                text("""
                SELECT
                    'daily_bars' as table_name,
                    MAX(ts) as latest_date,
                    COUNT(DISTINCT symbol) as symbol_count
                FROM daily_bars
                WHERE ts >= CURRENT_DATE - INTERVAL '7 days'
                UNION ALL
                SELECT
                    'fundamentals',
                    MAX(as_of),
                    COUNT(DISTINCT symbol)
                FROM fundamentals
                WHERE as_of >= CURRENT_DATE - INTERVAL '90 days'
                """)
            ).fetchall()

            health_status['checks']['data_freshness'] = {
                row.table_name: {
                    'latest_date': str(row.latest_date),
                    'symbol_count': row.symbol_count
                } for row in latest_data
            }

        log.info(f"Infrastructure health check completed: {health_status['overall_status']}")

    except Exception as e:
        log.error(f"Health check failed: {e}")
        health_status['overall_status'] = 'ERROR'
        health_status['checks']['error'] = str(e)

    return health_status
