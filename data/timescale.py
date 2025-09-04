from __future__ import annotations
import logging
from sqlalchemy import text, exc
from sqlalchemy.engine import Connection
from db import engine
from config import ENABLE_TIMESCALEDB, TIMESCALEDB_CHUNK_TIME_INTERVAL

log = logging.getLogger(__name__)

def is_timescaledb_available(conn: Connection) -> bool:
    try:
        result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")).scalar()
        return result is not None
    except Exception as e:
        log.warning(f"Could not check TimescaleDB availability: {e}")
        return False

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
        chk = conn.execute(text("SELECT 1 FROM timescaledb_information.hypertables WHERE table_name = 'daily_bars'")).scalar()
        if chk: return True
    except Exception:
        pass
    sql = f"""
        SELECT create_hypertable('daily_bars', 'ts', chunk_time_interval => INTERVAL '{TIMESCALEDB_CHUNK_TIME_INTERVAL}', if_not_exists => TRUE)
    """
    return _execute_optional_sql(conn, sql, "Convert daily_bars to hypertable")

def setup_timescaledb_policies(conn: Connection) -> bool:
    if not ENABLE_TIMESCALEDB or not is_timescaledb_available(conn):
        return False
    ok = True
    sql_comp = "SELECT add_compression_policy('daily_bars', INTERVAL '30 days', if_not_exists => TRUE)"
    if not _execute_optional_sql(conn, sql_comp, "Add compression policy"):
        ok = False
    sql_cagg = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS daily_bars_monthly
        WITH (timescaledb.continuous) AS
        SELECT symbol, time_bucket('1 month', ts) AS month,
               first(open, ts) as month_open, max(high) as month_high,
               min(low) as month_low, last(close, ts) as month_close,
               avg(volume) as avg_volume, sum(volume) as total_volume
        FROM daily_bars GROUP BY symbol, month
    """
    cagg_ok = _execute_optional_sql(conn, sql_cagg, "Create continuous aggregate")
    if not cagg_ok:
        ok = False
    if cagg_ok:
        sql_refresh = """
            SELECT add_continuous_aggregate_policy('daily_bars_monthly',
                start_offset => INTERVAL '3 months', end_offset => INTERVAL '1 day',
                schedule_interval => INTERVAL '1 day', if_not_exists => TRUE)
        """
        if not _execute_optional_sql(conn, sql_refresh, "Add refresh policy"):
            ok = False
    return ok

def setup_timescaledb() -> bool:
    if not ENABLE_TIMESCALEDB or not engine:
        return False
    success = True
    try:
        with engine.begin() as conn:
            if not enable_timescaledb_extension(conn):
                return False
            if not convert_daily_bars_to_hypertable(conn):
                success = False
            if not setup_timescaledb_policies(conn):
                success = False
    except Exception as e:
        log.error(f"TimescaleDB setup error: {e}", exc_info=True)
        success = False
    if success:
        log.info("TimescaleDB setup completed successfully")
    else:
        log.warning("TimescaleDB setup completed with errors")
    return success
