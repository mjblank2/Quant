"""
TimescaleDB integration for high-performance time-series data management.

TimescaleDB is a PostgreSQL extension that provides optimized storage and querying
for time-series data. This module handles the setup and optimization of daily_bars
as a TimescaleDB hypertable.
"""
from __future__ import annotations
import logging
from typing import Optional
from sqlalchemy import text
from db import engine
from config import ENABLE_TIMESCALEDB, TIMESCALEDB_CHUNK_TIME_INTERVAL

log = logging.getLogger(__name__)

def is_timescaledb_available() -> bool:
    """Check if TimescaleDB extension is available in the database."""
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")
            ).scalar()
            return result is not None
    except Exception as e:
        log.warning(f"Could not check TimescaleDB availability: {e}")
        return False

def enable_timescaledb_extension():
    """Enable TimescaleDB extension in the database."""
    if not ENABLE_TIMESCALEDB:
        log.info("TimescaleDB integration disabled by configuration")
        return False
    
    try:
        with engine.begin() as conn:
            # Check if extension is already enabled
            if is_timescaledb_available():
                log.info("TimescaleDB extension already enabled")
                return True
            
            # Enable the extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
            log.info("TimescaleDB extension enabled successfully")
            return True
            
    except Exception as e:
        log.error(f"Failed to enable TimescaleDB extension: {e}")
        log.info("Continuing with regular PostgreSQL tables")
        return False

def convert_daily_bars_to_hypertable() -> bool:
    """Convert daily_bars table to TimescaleDB hypertable for better performance."""
    if not ENABLE_TIMESCALEDB or not is_timescaledb_available():
        log.info("TimescaleDB not available, skipping hypertable conversion")
        return False
    
    try:
        with engine.begin() as conn:
            # Check if daily_bars is already a hypertable
            check_hypertable = conn.execute(
                text("""
                SELECT 1 FROM timescaledb_information.hypertables 
                WHERE table_name = 'daily_bars'
                """)
            ).scalar()
            
            if check_hypertable:
                log.info("daily_bars is already a hypertable")
                return True
            
            # Convert to hypertable
            log.info("Converting daily_bars to TimescaleDB hypertable")
            conn.execute(
                text(f"""
                SELECT create_hypertable(
                    'daily_bars', 
                    'ts',
                    chunk_time_interval => INTERVAL '{TIMESCALEDB_CHUNK_TIME_INTERVAL}',
                    if_not_exists => TRUE
                )
                """)
            )
            
            log.info("Successfully converted daily_bars to hypertable")
            return True
            
    except Exception as e:
        log.error(f"Failed to convert daily_bars to hypertable: {e}")
        return False

def setup_timescaledb_policies():
    """Set up compression and retention policies for optimal performance."""
    if not ENABLE_TIMESCALEDB or not is_timescaledb_available():
        return False
    
    try:
        with engine.begin() as conn:
            # Add compression policy for data older than 30 days
            # This significantly reduces storage space for historical data
            conn.execute(
                text("""
                SELECT add_compression_policy(
                    'daily_bars', 
                    INTERVAL '30 days',
                    if_not_exists => TRUE
                )
                """)
            )
            
            # Add continuous aggregate for monthly OHLCV data
            # This creates pre-computed monthly summaries for faster queries
            conn.execute(
                text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS daily_bars_monthly
                WITH (timescaledb.continuous) AS
                SELECT 
                    symbol,
                    time_bucket('1 month', ts) AS month,
                    first(open, ts) as month_open,
                    max(high) as month_high,
                    min(low) as month_low,
                    last(close, ts) as month_close,
                    avg(volume) as avg_volume,
                    sum(volume) as total_volume,
                    count(*) as trading_days
                FROM daily_bars
                GROUP BY symbol, month
                """)
            )
            
            # Add refresh policy for the continuous aggregate
            conn.execute(
                text("""
                SELECT add_continuous_aggregate_policy(
                    'daily_bars_monthly',
                    start_offset => INTERVAL '3 months',
                    end_offset => INTERVAL '1 day',
                    schedule_interval => INTERVAL '1 day',
                    if_not_exists => TRUE
                )
                """)
            )
            
            log.info("TimescaleDB policies set up successfully")
            return True
            
    except Exception as e:
        log.error(f"Failed to set up TimescaleDB policies: {e}")
        return False

def optimize_timescaledb_indexes():
    """Create optimized indexes for common query patterns."""
    if not ENABLE_TIMESCALEDB or not is_timescaledb_available():
        return False
    
    try:
        with engine.begin() as conn:
            # Create index for symbol-based queries with time ordering
            conn.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_daily_bars_symbol_ts_covering 
                ON daily_bars (symbol, ts DESC) 
                INCLUDE (close, volume, adj_close)
                """)
            )
            
            # Create index for volume-based screening queries
            conn.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_daily_bars_volume_recent
                ON daily_bars (volume DESC, ts DESC)
                WHERE ts >= CURRENT_DATE - INTERVAL '90 days'
                """)
            )
            
            log.info("TimescaleDB indexes optimized")
            return True
            
    except Exception as e:
        log.error(f"Failed to optimize TimescaleDB indexes: {e}")
        return False

def get_timescaledb_info() -> dict:
    """Get information about TimescaleDB setup and performance."""
    info = {
        'enabled': ENABLE_TIMESCALEDB,
        'available': is_timescaledb_available(),
        'hypertable_configured': False,
        'compression_enabled': False,
        'chunk_count': 0,
        'compressed_chunks': 0
    }
    
    if not info['available']:
        return info
    
    try:
        with engine.connect() as conn:
            # Check hypertable status
            hypertable_check = conn.execute(
                text("""
                SELECT 1 FROM timescaledb_information.hypertables 
                WHERE table_name = 'daily_bars'
                """)
            ).scalar()
            info['hypertable_configured'] = hypertable_check is not None
            
            if info['hypertable_configured']:
                # Get chunk information
                chunk_info = conn.execute(
                    text("""
                    SELECT 
                        COUNT(*) as total_chunks,
                        COUNT(*) FILTER (WHERE is_compressed) as compressed_chunks
                    FROM timescaledb_information.chunks
                    WHERE hypertable_name = 'daily_bars'
                    """)
                ).fetchone()
                
                if chunk_info:
                    info['chunk_count'] = chunk_info.total_chunks
                    info['compressed_chunks'] = chunk_info.compressed_chunks
                    info['compression_enabled'] = chunk_info.compressed_chunks > 0
                    
    except Exception as e:
        log.warning(f"Could not retrieve TimescaleDB info: {e}")
    
    return info

def setup_timescaledb():
    """Complete TimescaleDB setup process."""
    log.info("Starting TimescaleDB setup")
    
    if not ENABLE_TIMESCALEDB:
        log.info("TimescaleDB disabled in configuration")
        return False
    
    success = True
    
    # Enable extension
    if not enable_timescaledb_extension():
        return False
    
    # Convert daily_bars to hypertable
    if not convert_daily_bars_to_hypertable():
        success = False
    
    # Set up policies
    if not setup_timescaledb_policies():
        success = False
    
    # Optimize indexes
    if not optimize_timescaledb_indexes():
        success = False
    
    if success:
        log.info("TimescaleDB setup completed successfully")
    else:
        log.warning("TimescaleDB setup completed with some errors")
    
    return success