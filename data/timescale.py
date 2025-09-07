
from __future__ import annotations

import logging
from sqlalchemy import text, exc
from sqlalchemy.engine import Connection, Engine
from typing import Optional

try:
    from db import engine  # type: ignore
except Exception:
    engine = None

try:
    from config import ENABLE_TIMESCALEDB, TIMESCALEDB_CHUNK_TIME_INTERVAL
except Exception:
    ENABLE_TIMESCALEDB = False
    TIMESCALEDB_CHUNK_TIME_INTERVAL = "1 month"

log = logging.getLogger("data.timescale")


def is_timescaledb_available(conn: Optional[Connection] = None) -> bool:
    """Check if TimescaleDB extension is installed."""
    if conn is None:
        if engine is None:
            return False
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")).scalar()
        except Exception as e:
            log.warning(f"Could not check TimescaleDB availability: {e}")
            return False
    else:
        try:
            result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")).scalar()
        except Exception as e:
            log.warning(f"Could not check TimescaleDB availability: {e}")
            return False
    return result is not None

def _execute_optional_sql(conn: Connection, sql: str, description: str) -> bool:
    log.info(f"Attempting (Optional): {description}")
    try:
        with conn.begin_nested():
            conn.execute(text(sql))
        log.info(f"Success: {description}")
        return True
    except (exc.SQLAlchemyError, Exception) as e:
        log.warning(f"Failed (Optional): {description}. Error: {e}")
        return False

def enable_timescaledb_extension(conn: Connection) -> bool:
    if not ENABLE_TIMESCALEDB:
        return False
    if is_timescaledb_available(conn):
        return True
    return _execute_optional_sql(conn, "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;", "Enable TimescaleDB extension")

def convert_daily_bars_to_hypertable(conn: Connection) -> bool:
    if not ENABLE_TIMESCALEDB or not is_timescaledb_available(conn):
        return False
    try:
        check = conn.execute(text("SELECT 1 FROM timescaledb_information.hypertables WHERE table_name = 'daily_bars'")).scalar()
        if check: return True
    except Exception:
        pass
    sql = f"""
        SELECT create_hypertable(
            'daily_bars', 'ts',
            chunk_time_interval => INTERVAL '{TIMESCALEDB_CHUNK_TIME_INTERVAL}',
            if_not_exists => TRUE
        )
    """
    return _execute_optional_sql(conn, sql, "Convert daily_bars to hypertable")

def setup_timescaledb_policies(conn: Connection) -> bool:
    if not ENABLE_TIMESCALEDB or not is_timescaledb_available(conn):
        return False
    success = True
    sql_comp = "SELECT add_compression_policy('daily_bars', INTERVAL '30 days', if_not_exists => TRUE)"
    if not _execute_optional_sql(conn, sql_comp, "Add compression policy"):
        success = False
    sql_cagg = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS daily_bars_monthly
        WITH (timescaledb.continuous) AS
        SELECT symbol, time_bucket('1 month', ts) AS month,
               first(open, ts) as month_open, max(high) as month_high,
               min(low) as month_low, last(close, ts) as month_close,
               avg(volume) as avg_volume, sum(volume) as total_volume
        FROM daily_bars GROUP BY symbol, month
    """
    cagg_success = _execute_optional_sql(conn, sql_cagg, "Create continuous aggregate")
    if not cagg_success: success = False
    if cagg_success:
        sql_refresh = """
            SELECT add_continuous_aggregate_policy(
                'daily_bars_monthly', start_offset => INTERVAL '3 months', end_offset => INTERVAL '1 day',
                schedule_interval => INTERVAL '1 day', if_not_exists => TRUE
            )
        """
        if not _execute_optional_sql(conn, sql_refresh, "Add refresh policy"):
            success = False
    return success

def optimize_timescaledb_indexes(conn: Connection) -> bool:
    # Add any optional index/comment statements here.
    return True

def setup_timescaledb(db_engine: Optional[Engine] = None) -> bool:
    """Configure TimescaleDB if enabled."""
    if db_engine is None:
        db_engine = engine
    if not ENABLE_TIMESCALEDB or db_engine is None:
        return False
    success = True
    try:
        with db_engine.begin() as conn:
            if not enable_timescaledb_extension(conn):
                return False
            if not convert_daily_bars_to_hypertable(conn):
                success = False
            if not setup_timescaledb_policies(conn):
                success = False
            if not optimize_timescaledb_indexes(conn):
                success = False
    except Exception as e:
        log.error(f"An unexpected error occurred during TimescaleDB setup: {e}", exc_info=True)
        success = False
    if success:
        log.info("TimescaleDB setup completed successfully")
    else:
        log.warning("TimescaleDB setup completed with errors. System will operate without full optimization.")
    return success


def get_timescaledb_info() -> dict:
    """Return basic TimescaleDB status information."""
    info = {
        "enabled": ENABLE_TIMESCALEDB,
        "available": False,
        "hypertable_configured": False,
        "compression_enabled": False,
        "chunk_count": 0,
        "compressed_chunks": 0,
    }
    if not ENABLE_TIMESCALEDB:
        return info

    info["available"] = is_timescaledb_available()
    return info
