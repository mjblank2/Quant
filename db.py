from __future__ import annotations
from contextlib import nullcontext
from sqlalchemy import create_engine, String, Date, DateTime, Integer, Float, Boolean, BigInteger, Index, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.types import JSON
from datetime import datetime, date
import pandas as pd
import numpy as np
from config import DATABASE_URL
import logging

log = logging.getLogger(__name__)

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required.")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)

class Base(DeclarativeBase):
    pass

# --- Core Market Data ---
class DailyBar(Base):
    __tablename__ = "daily_bars"
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    ts: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[int] = mapped_column(BigInteger)
    vwap: Mapped[float | None] = mapped_column(Float, nullable=True)
    trade_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    __table_args__ = (Index("ix_daily_bars_symbol_ts", "symbol", "ts"),)

class Universe(Base):
    __tablename__ = "universe"
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    exchange: Mapped[str | None] = mapped_column(String(12), nullable=True)
    market_cap: Mapped[float | None] = mapped_column(Float, nullable=True)  # added to match migrations & code usage
    adv_usd_20: Mapped[float | None] = mapped_column(Float, nullable=True)
    included: Mapped[bool] = mapped_column(Boolean, default=True)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

# v17: universe snapshots for survivorship-safe training
class UniverseHistory(Base):
    __tablename__ = "universe_history"
    as_of: Mapped[date] = mapped_column(Date, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    __table_args__ = (Index("ix_universe_hist_asof", "as_of"), Index("ix_universe_hist_symbol", "symbol"),)

# PIT fundamentals + shares
class SharesOutstanding(Base):
    __tablename__ = "shares_outstanding"
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    as_of: Mapped[date] = mapped_column(Date, primary_key=True)
    shares: Mapped[int] = mapped_column(BigInteger)
    source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    knowledge_date: Mapped[date | None] = mapped_column(Date, nullable=True)  # bi-temporal support
    __table_args__ = (
        Index("ix_shares_symbol_asof", "symbol", "as_of"),
        Index("ix_shares_outstanding_bitemporal", "symbol", "as_of", "knowledge_date"),
    )

class Fundamentals(Base):
    __tablename__ = "fundamentals"
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    as_of: Mapped[date] = mapped_column(Date, primary_key=True)
    pe_ttm: Mapped[float | None] = mapped_column(Float, nullable=True)
    pb: Mapped[float | None] = mapped_column(Float, nullable=True)
    ps_ttm: Mapped[float | None] = mapped_column(Float, nullable=True)
    debt_to_equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_on_assets: Mapped[float | None] = mapped_column(Float, nullable=True)
    gross_margins: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_margins: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    available_at: Mapped[date | None] = mapped_column(Date, nullable=True)  # when data became available
    knowledge_date: Mapped[date | None] = mapped_column(Date, nullable=True)  # bi-temporal support
    __table_args__ = (
        Index("ix_fundamentals_symbol_asof", "symbol", "as_of"),
        Index("ix_fundamentals_bitemporal", "symbol", "as_of", "knowledge_date"),
    )

class AltSignal(Base):
    __tablename__ = "alt_signals"
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    ts: Mapped[date] = mapped_column(Date, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    __table_args__ = (Index("ix_alt_symbol_ts", "symbol", "ts"),)

# v17: Russell membership events
class RussellMembership(Base):
    __tablename__ = "russell_membership"
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    ts: Mapped[date] = mapped_column(Date, primary_key=True)
    action: Mapped[str] = mapped_column(String(8))  # add|drop|keep
    __table_args__ = (Index("ix_russell_ts", "ts"),)

# --- Modeling Artifacts ---
class Feature(Base):
    __tablename__ = "features"
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    ts: Mapped[date] = mapped_column(Date, primary_key=True)
    ret_1d: Mapped[float | None] = mapped_column(Float)
    ret_5d: Mapped[float | None] = mapped_column(Float)
    ret_21d: Mapped[float | None] = mapped_column(Float)
    mom_21: Mapped[float | None] = mapped_column(Float)
    mom_63: Mapped[float | None] = mapped_column(Float)
    vol_21: Mapped[float | None] = mapped_column(Float)
    rsi_14: Mapped[float | None] = mapped_column(Float)
    turnover_21: Mapped[float | None] = mapped_column(Float)
    size_ln: Mapped[float | None] = mapped_column(Float)
    adv_usd_21: Mapped[float | None] = mapped_column(Float)  # v16 extension
    reversal_5d_z: Mapped[float | None] = mapped_column(Float, nullable=True)
    ivol_63: Mapped[float | None] = mapped_column(Float, nullable=True)
    beta_63: Mapped[float | None] = mapped_column(Float, nullable=True)  # v16 extension
    overnight_gap: Mapped[float | None] = mapped_column(Float, nullable=True)  # v16 extension
    illiq_21: Mapped[float | None] = mapped_column(Float, nullable=True)  # v16 extension
    fwd_ret: Mapped[float | None] = mapped_column(Float, nullable=True)
    fwd_ret_resid: Mapped[float | None] = mapped_column(Float, nullable=True)
    pead_event: Mapped[float | None] = mapped_column(Float, nullable=True)
    pead_surprise_eps: Mapped[float | None] = mapped_column(Float, nullable=True)
    pead_surprise_rev: Mapped[float | None] = mapped_column(Float, nullable=True)
    russell_inout: Mapped[float | None] = mapped_column(Float, nullable=True)
    f_pe_ttm: Mapped[float | None] = mapped_column(Float, nullable=True)
    f_pb: Mapped[float | None] = mapped_column(Float, nullable=True)
    f_ps_ttm: Mapped[float | None] = mapped_column(Float, nullable=True)
    f_debt_to_equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    f_roa: Mapped[float | None] = mapped_column(Float, nullable=True)
    f_gm: Mapped[float | None] = mapped_column(Float, nullable=True)
    f_profit_margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    f_current_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    __table_args__ = (Index("ix_features_symbol_ts", "symbol", "ts"),)

class Prediction(Base):
    __tablename__ = "predictions"
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    ts: Mapped[date] = mapped_column(Date, primary_key=True)
    model_version: Mapped[str] = mapped_column(String(32), primary_key=True, default="xgb_v1")
    horizon: Mapped[int] = mapped_column(Integer, default=5)
    y_pred: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        Index("ix_predictions_ts", "ts"),
        Index("ix_predictions_symbol_ts", "symbol", "ts"),
        Index("ix_predictions_ts_model", "ts", "model_version"),
    )

# --- OMS and State ---
class TargetPosition(Base):
    __tablename__ = "target_positions"
    ts: Mapped[date] = mapped_column(Date, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    weight: Mapped[float] = mapped_column(Float)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_shares: Mapped[int | None] = mapped_column(Integer, nullable=True)
    __table_args__ = (Index("ix_target_positions_ts_symbol", "ts", "symbol"),)

class CurrentPosition(Base):
    __tablename__ = "current_positions"
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    shares: Mapped[int] = mapped_column(Integer)
    market_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_basis: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SystemState(Base):
    __tablename__ = "system_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    nav: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    last_reconciled: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

class Trade(Base):
    __tablename__ = "trades"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    trade_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[str] = mapped_column(String(4))
    quantity: Mapped[int] = mapped_column(Integer)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="generated")
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    client_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)  # added to match migrations and usage
    filled_quantity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    order_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # v17: lot hints, overlay notes

# v17: borrow & options overlay tables
class ShortBorrow(Base):
    __tablename__ = "short_borrow"
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    ts: Mapped[date] = mapped_column(Date, primary_key=True)
    available: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fee_bps: Mapped[float | None] = mapped_column(Float, nullable=True)
    short_interest: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    __table_args__ = (Index("ix_borrow_symbol_ts", "symbol", "ts"),)

class OptionOverlay(Base):
    __tablename__ = "option_overlays"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    as_of: Mapped[date] = mapped_column(Date, index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    strategy: Mapped[str] = mapped_column(String(16))  # protective_put | put_spread | collar
    tenor_days: Mapped[int] = mapped_column(Integer)
    put_strike: Mapped[float | None] = mapped_column(Float, nullable=True)
    call_strike: Mapped[float | None] = mapped_column(Float, nullable=True)
    est_premium_pct: Mapped[float | None] = mapped_column(Float, nullable=True)  # of notional
    notes: Mapped[str | None] = mapped_column(String(256), nullable=True)

class BacktestEquity(Base):
    __tablename__ = "backtest_equity"
    ts: Mapped[date] = mapped_column(Date, primary_key=True)
    equity: Mapped[float] = mapped_column(Float)
    daily_return: Mapped[float] = mapped_column(Float)
    drawdown: Mapped[float] = mapped_column(Float)
    tcost_impact: Mapped[float] = mapped_column(Float, default=0.0)

# --- Task Queue Status Tracking ---
class TaskStatus(Base):
    __tablename__ = "task_status"
    task_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # PENDING, STARTED, SUCCESS, FAILURE, RETRY
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)  # 0-100
    __table_args__ = (Index("ix_task_status_created_at", "created_at"),)

# --- Data Quality and Governance ---
class DataValidationLog(Base):
    __tablename__ = "data_validation_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    validation_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # PASSED, FAILED, WARNING
    message: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    affected_symbols: Mapped[list | None] = mapped_column(JSON, nullable=True)
    __table_args__ = (
        Index("ix_validation_log_timestamp", "run_timestamp"),
        Index("ix_validation_log_type_status", "validation_type", "status"),
    )


class DataLineage(Base):
    __tablename__ = "data_lineage"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    table_name: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    data_date: Mapped[date] = mapped_column(Date, nullable=False)
    ingestion_timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_timestamp: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    lineage_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    __table_args__ = (
        Index("ix_lineage_table_symbol_date", "table_name", "symbol", "data_date"),
    )

def create_tables():
    Base.metadata.create_all(engine)
    from config import STARTING_CAPITAL
    try:
        with engine.begin() as conn:
            # init system_state if missing
            exists = conn.execute(text("SELECT 1 FROM system_state WHERE id=1")).scalar()
            if not exists:
                conn.execute(text("INSERT INTO system_state (id, nav, cash) VALUES (1, :nav, :cash)"),
                             {"nav": STARTING_CAPITAL, "cash": STARTING_CAPITAL})
    except Exception as e:
        log.warning(f"Could not initialize SystemState: {e}")


# Efficient UPSERT helper

# Cache for table column information to avoid repeated database queries
_table_columns_cache = {}

# Track warnings to avoid repeated logging of the same column issues
_column_warning_cache = {}

# Track failed cache refresh attempts to avoid repeated refresh attempts
_cache_refresh_failures = {}

def clear_table_columns_cache():
    """Clear the table columns cache. Useful after database migrations."""
    _table_columns_cache.clear()
    _column_warning_cache.clear()
    _cache_refresh_failures.clear()


def _max_bind_params_for_connection(connection) -> int:
    """
    Get the maximum number of bind parameters for the given database connection.
    Returns conservative limits to avoid parameter limit errors.
    """
    try:
        url_str = str(connection.engine.url).lower()

        if "sqlite" in url_str:
            # SQLite default limit is 999 variables; use full limit minus small safety margin
            return 999
        elif "postgresql" in url_str:
            # PostgreSQL varies by configuration, default is often 32767
            # But some hosted services may have lower limits, use conservative value
            return 16000
        elif "mysql" in url_str:
            # MySQL limit is typically 65535
            return 32000
        else:
            # Unknown database, use very conservative default
            return 1000

    except Exception as e:
        log.warning(f"Could not determine database type for parameter limits: {e}")
        return 1000  # Very conservative fallback


def _should_log_column_warning(table_name: str, missing_columns: set) -> bool:
    """
    Check if we should log a warning about missing columns.
    Rate-limits repeated warnings about the same missing columns for the same table.
    Implements aggressive rate limiting for known issues to prevent memory problems.
    """
    import time

    warning_key = (table_name, frozenset(missing_columns))
    current_time = time.time()

    # Special handling for adj_close column warnings - these are often expected
    # during data ingestion and should be rate-limited more aggressively
    if 'adj_close' in missing_columns and len(missing_columns) == 1:
        rate_limit_seconds = 300  # 5 minutes for adj_close only warnings
    else:
        rate_limit_seconds = 60   # 1 minute for other column warnings

    # Check if we've already logged this warning recently
    if warning_key in _column_warning_cache:
        last_logged = _column_warning_cache[warning_key]
        if current_time - last_logged < rate_limit_seconds:
            return False

    # Implement cache size management to prevent memory growth
    # Remove old entries if cache gets too large
    MAX_CACHE_SIZE = 100
    if len(_column_warning_cache) > MAX_CACHE_SIZE:
        # Remove entries older than 1 hour to free memory
        cutoff_time = current_time - 3600
        old_keys = [k for k, v in _column_warning_cache.items() if v < cutoff_time]
        for k in old_keys:
            del _column_warning_cache[k]

        # If still too large after cleanup, remove oldest entries
        if len(_column_warning_cache) > MAX_CACHE_SIZE:
            sorted_items = sorted(_column_warning_cache.items(), key=lambda x: x[1])
            items_to_remove = len(_column_warning_cache) - MAX_CACHE_SIZE + 10
            for k, _ in sorted_items[:items_to_remove]:
                del _column_warning_cache[k]

    _column_warning_cache[warning_key] = current_time
    return True


def _get_table_columns(connection, table):
    """Get table columns with caching to avoid repeated database queries"""
    table_name = table.__tablename__
    if table_name in _table_columns_cache:
        return _table_columns_cache[table_name]

    try:
        # Get actual table columns from database
        if "postgresql" in str(connection.engine.url).lower():
            # PostgreSQL - try multiple approaches to find table columns
            actual_columns = set()

            # First try: look in current schema (don't specify schema)
            try:
                actual_columns_result = connection.execute(text("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = :table_name
                    AND table_schema = current_schema()
                """), {"table_name": table_name})
                actual_columns = {row[0] for row in actual_columns_result.fetchall()}
            except Exception:
                pass

            # Second try: look in 'public' schema (original approach)
            if not actual_columns:
                try:
                    actual_columns_result = connection.execute(text("""
                        SELECT column_name FROM information_schema.columns
                        WHERE table_name = :table_name AND table_schema = 'public'
                    """), {"table_name": table_name})
                    actual_columns = {row[0] for row in actual_columns_result.fetchall()}
                except Exception:
                    pass

            # Third try: look in any schema (remove schema restriction)
            if not actual_columns:
                try:
                    actual_columns_result = connection.execute(text("""
                        SELECT column_name FROM information_schema.columns
                        WHERE table_name = :table_name
                    """), {"table_name": table_name})
                    actual_columns = {row[0] for row in actual_columns_result.fetchall()}
                except Exception:
                    pass

        else:
            # SQLite
            actual_columns_result = connection.execute(text(f"PRAGMA table_info({table_name})"))
            actual_columns = {row[1] for row in actual_columns_result.fetchall()}  # column name is index 1 in SQLite

        # Cache the result only if we found columns
        if actual_columns:
            _table_columns_cache[table_name] = actual_columns
            return actual_columns
        else:
            # If no columns found, don't cache and return None
            log.debug(f"No columns found for table {table_name} in database schema inspection")
            return None

    except Exception as e:
        log.warning(f"Could not inspect table columns for {table_name}: {e}. Using all DataFrame columns.")
        # Don't cache failed attempts
        return None


def upsert_dataframe(df: pd.DataFrame, table, conflict_cols: list[str], chunk_size: int = 50000, conn=None):
    """
    Insert/update DataFrame rows into 'table' with ON CONFLICT handling.
    Automatically batches statements to stay under PostgreSQL's parameter limit.
    """
    if df is None or df.empty:
        return
    df = df.replace({pd.NA: None, np.nan: None})

    # Proactively drop duplicates on the conflict key within this batch to avoid
    # Postgres "ON CONFLICT DO UPDATE command cannot affect row a second time".
    # Only do this if all conflict columns are present in the DataFrame.
    if conflict_cols and set(conflict_cols).issubset(df.columns):
        before = len(df)
        # Keep the last occurrence so the most recent values win
        df = df.drop_duplicates(subset=conflict_cols, keep='last').reset_index(drop=True)
        removed = before - len(df)
        if removed > 0:
            log.debug(f"Removed {removed} duplicate rows based on conflict columns {conflict_cols} before UPSERT")

    # Use dynamic parameter limits based on the database connection type
    # This prevents parameter limit errors across different database backends

    ctx = engine.begin() if conn is None else nullcontext(conn)
    with ctx as connection:
        # Filter DataFrame columns to only include columns that exist in the actual database table
        # This prevents errors when migrations haven't been applied yet
        actual_columns = _get_table_columns(connection, table)
        if actual_columns is None:
            # If column inspection failed, use all DataFrame columns
            actual_columns = set(df.columns)

        df_columns = set(df.columns)
        valid_columns = df_columns.intersection(actual_columns)

        if valid_columns != df_columns:
            missing_in_table = df_columns - actual_columns

            # If columns are missing, try refreshing the cache in case schema has been updated
            # This handles cases where migrations have been applied but cache is stale
            import time

            table_name = table.__tablename__
            refresh_key = (table_name, frozenset(missing_in_table))
            current_time = time.time()

            # Avoid repeated cache refresh attempts for the same missing columns
            should_refresh = table_name in _table_columns_cache
            if refresh_key in _cache_refresh_failures:
                last_failed_refresh = _cache_refresh_failures[refresh_key]
                # Only retry cache refresh after 10 minutes
                should_refresh = should_refresh and (current_time - last_failed_refresh > 600)

            if should_refresh:
                log.debug(f"Detected missing columns {missing_in_table} in {table_name}, refreshing column cache")
                # Clear this table from cache and re-fetch
                del _table_columns_cache[table_name]
                actual_columns_refreshed = _get_table_columns(connection, table)

                if actual_columns_refreshed is not None:
                    actual_columns = actual_columns_refreshed
                    valid_columns = df_columns.intersection(actual_columns)
                    missing_in_table_after_refresh = df_columns - actual_columns

                    # If columns are still missing after refresh, record this to avoid repeated attempts
                    if missing_in_table_after_refresh == missing_in_table:
                        _cache_refresh_failures[refresh_key] = current_time

                    missing_in_table = missing_in_table_after_refresh
                    log.debug(f"Cache refresh completed for {table_name}, found {len(actual_columns)} columns")
                else:
                    log.warning(f"Failed to refresh column cache for {table_name}")
                    _cache_refresh_failures[refresh_key] = current_time
            else:
                # Table not in cache yet, this is the first access - no need to refresh
                if table_name not in _table_columns_cache:
                    log.debug(f"Table {table_name} not in cache yet, first access")
                else:
                    log.debug(f"Skipping cache refresh for {table_name} - recently attempted for same missing columns")

            # Only warn and drop columns if they're still missing after cache refresh
            if df_columns != valid_columns:
                # Use enhanced warning deduplication
                try:
                    from warning_dedup import warn_adj_close_missing, warn_column_mismatch

                    if 'adj_close' in missing_in_table and len(missing_in_table) == 1:
                        warn_adj_close_missing(log, table.__tablename__)
                    else:
                        warn_column_mismatch(log, table.__tablename__, list(missing_in_table))
                except ImportError:
                    # Fallback to rate-limited warnings if dedup module not available
                    if _should_log_column_warning(table.__tablename__, missing_in_table):
                        if 'adj_close' in missing_in_table and len(missing_in_table) == 1:
                            log.debug(f"Dropping adj_close column (not present in {table.__tablename__}). This may indicate pending migration.")
                        else:
                            log.warning(f"Dropping columns not present in {table.__tablename__}: {missing_in_table}")
                df = df[list(valid_columns)]

        if df.empty:
            return

        # Proactive deduplication to prevent CardinalityViolation errors
        # Remove any duplicate rows based on conflict columns before any INSERT attempt
        if conflict_cols and set(conflict_cols).issubset(df.columns):
            original_size = len(df)
            df = df.drop_duplicates(subset=conflict_cols, keep='last').reset_index(drop=True)
            dedupe_size = len(df)
            if dedupe_size < original_size:
                # Use DEBUG level since this is expected behavior during normal data processing
                log.debug(f"Proactively removed {original_size - dedupe_size} duplicate rows to prevent CardinalityViolation on conflict cols {conflict_cols}")

        cols_all = list(df.columns)
        # Get dynamic parameter limits based on connection type
        max_bind_params = _max_bind_params_for_connection(connection)
        # rows per statement bounded by max_bind_params / num_columns
        # Add additional safety: ensure we don't exceed reasonable batch sizes
        theoretical_max_rows = max_bind_params // max(1, len(cols_all))
        per_stmt_rows = max(1, min(chunk_size, theoretical_max_rows, 1000))  # Cap at 1000 rows max

        for start in range(0, len(df), per_stmt_rows):
            part = df.iloc[start:start + per_stmt_rows]

            # Additional safety: deduplicate each chunk to prevent cardinality violations
            # This handles edge cases where chunking might create duplicates
            if conflict_cols and set(conflict_cols).issubset(part.columns):
                chunk_original_size = len(part)
                part = part.drop_duplicates(subset=conflict_cols, keep='last').reset_index(drop=True)
                chunk_dedupe_size = len(part)
                if chunk_dedupe_size < chunk_original_size:
                    log.debug(f"Chunk deduplication: removed {chunk_original_size - chunk_dedupe_size} duplicates from batch")

            cols = list(part.columns)
            records = part.to_dict(orient="records")
            if not records:
                continue

            try:
                stmt = insert(table).values(records)
                update_cols = {c: getattr(stmt.excluded, c) for c in cols if c not in conflict_cols}
                if update_cols:
                    stmt = stmt.on_conflict_do_update(index_elements=conflict_cols, set_=update_cols)
                else:
                    stmt = stmt.on_conflict_do_nothing(index_elements=conflict_cols)
                connection.execute(stmt)
            except Exception as e:
                msg = str(e).lower()
                # More specific parameter limit detection to avoid false positives
                is_param_limit = ("too many variables" in msg) or ("too many sql variables" in msg) or ("bind parameter" in msg and "limit" in msg)
                is_txn_abort = ("infailedsqltransaction" in msg) or ("cannot operate on closed transaction" in msg)
                is_cardinality = ("cardinalityviolation" in msg) or ("on conflict do update command cannot affect row a second time" in msg)

                # Don't treat readonly database errors as parameter limits
                is_readonly = ("readonly database" in msg) or ("attempt to write a readonly database" in msg)

                if (is_param_limit or (is_txn_abort and not is_readonly) or is_cardinality) and len(records) > 1:
                    if is_param_limit:
                        log.warning(f"Parameter limit error with {len(records)} records, retrying with smaller batches")
                    elif is_txn_abort:
                        log.warning(f"Transaction abort error with {len(records)} records, retrying with smaller batches")
                    if is_cardinality:
                        log.warning(f"CardinalityViolation with {len(records)} records, deduping on conflict cols {conflict_cols} and "
                                    f"retrying in smaller batches")

                    # Rollback the current transaction to clear the aborted state
                    try:
                        connection.rollback()
                    except Exception:
                        pass  # Ignore rollback errors - transaction might already be rolled back

                    # Rebuild smaller DataFrame from records for retry
                    smaller_df = pd.DataFrame(records)

                    # Remove any duplicate rows based on the conflict columns before retry
                    if len(smaller_df) > 0 and conflict_cols and set(conflict_cols).issubset(smaller_df.columns):
                        original_size = len(smaller_df)
                        smaller_df = smaller_df.drop_duplicates(subset=conflict_cols, keep='last').reset_index(drop=True)
                        dedupe_size = len(smaller_df)

                        # Use enhanced warning deduplication for cardinality violations
                        if dedupe_size < original_size:
                            try:
                                from warning_dedup import warn_cardinality_violation
                                warn_cardinality_violation(log, table.__tablename__,
                                                           original_size - dedupe_size, original_size)
                            except ImportError:
                                # Fallback to simple logging
                                removed_count = original_size - dedupe_size
                                if removed_count > original_size * 0.5:  # More than 50% are duplicates
                                    log.warning(f"Removed {removed_count} duplicate rows during retry (significant duplication detected)")
                                else:
                                    log.debug(f"Removed {removed_count} duplicate rows during retry to prevent CardinalityViolation")
                        if dedupe_size < original_size:
                            # Use DEBUG level for retry deduplication since this is expected behavior
                            # Only log as WARNING if we're removing a significant number of duplicates
                            removed_count = original_size - dedupe_size
                            if removed_count > original_size * 0.5:  # More than 50% are duplicates
                                log.warning(f"Removed {removed_count} duplicate rows during retry (significant duplication detected)")
                            else:
                                log.debug(f"Removed {removed_count} duplicate rows during retry to prevent CardinalityViolation")

                    # Retry with smaller chunks using recursive call outside current transaction
                    # This handles the parameter limit case by starting fresh with a new transaction
                    log.debug(f"Retrying {len(smaller_df)} records in smaller chunks of 10 rows each")
                    upsert_dataframe(smaller_df, table, conflict_cols, chunk_size=10, conn=None)
                    return  # Exit after successful retry
                else:
                    # Re-raise if it's not a handled issue or if we're already at minimum size
                    raise
