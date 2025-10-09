from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from datetime import date
from typing import Any, Iterator, Optional, Sequence

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    Float,
    Integer,
    String,
    Index,
    Boolean,
    DateTime,
    create_engine,
    inspect,
    tuple_,
)
from sqlalchemy.engine import Connection
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


def _normalise_dsn(url: str) -> str:
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://") and "+psycopg" not in url:
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _create_engine_from_env() -> Any:
    """
    Create a SQLAlchemy engine from the ``DATABASE_URL`` environment variable.

    In addition to enabling ``pool_pre_ping`` (which tests connections before
    returning them to callers) this helper now also configures
    ``pool_recycle`` to proactively recycle TCP connections after a period of
    inactivity.  Without recycling, long‑lived connections can accumulate and
    eventually be closed by the PostgreSQL server or a network proxy,
    resulting in ``psycopg.OperationalError`` errors such as
    ``SSL SYSCALL error: EOF detected``.  Recycling idle connections
    mitigates these transient disconnections by ensuring that stale
    connections are not reused.

    Returns
    -------
    sqlalchemy.engine.Engine
        A SQLAlchemy engine configured for the application's database.
    """
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    dsn = _normalise_dsn(dsn)
    # Recycle connections after 10 minutes (600 seconds) of inactivity.  This
    # prevents stale connections from causing SSL EOF errors when the server
    # drops idle sessions.
    return create_engine(dsn, pool_pre_ping=True, pool_recycle=600)


engine = _create_engine_from_env()
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()
logger = logging.getLogger(__name__)

_PREDICTION_INDEXES: Sequence[Any]
if engine.dialect.name == "sqlite":
    _PREDICTION_INDEXES = ()
else:
    _PREDICTION_INDEXES = (
        Index("ix_predictions_ts", "ts"),
        Index("ix_predictions_ts_model", "ts", "model_version"),
    )

_table_columns_cache: dict[tuple[str, str], list[str]] = {}
_column_warning_cache: OrderedDict[tuple[str, tuple[str, ...]], float] = OrderedDict()
_MAX_COLUMN_CACHE_SIZE = 100
_ADJ_CLOSE_RATE_LIMIT = 300.0  # seconds
_GENERAL_RATE_LIMIT = 60.0


def _engine_cache_key(target: Any) -> str:
    """Generate a cache key for column lookups based on the engine URL."""

    try:
        if hasattr(target, "engine"):
            target = target.engine
        url = getattr(target, "url", None)
        if url is not None:
            return str(url)
        dialect = getattr(target, "dialect", None)
        if dialect is not None and getattr(dialect, "name", None):
            return str(dialect.name)
    except Exception:  # pragma: no cover - defensive for mocks
        pass

    try:
        return str(engine.url)
    except Exception:  # pragma: no cover - fallback when engine lacks URL
        return "default"


def clear_table_columns_cache() -> None:
    """Clear cached table column metadata."""

    _table_columns_cache.clear()
    _column_warning_cache.clear()


def _should_log_column_warning(table_name: str, missing_cols: set[str]) -> bool:
    """Determine whether a dropped-column warning should be emitted."""

    if not missing_cols:
        return False

    normalized_missing = tuple(sorted(missing_cols))
    cache_key = (table_name, normalized_missing)
    now = time.time()

    last_logged = _column_warning_cache.get(cache_key)

    is_adj_close_only = normalized_missing == ("adj_close",)
    rate_limit = _ADJ_CLOSE_RATE_LIMIT if is_adj_close_only else _GENERAL_RATE_LIMIT

    if last_logged is not None and now - last_logged < rate_limit:
        return False

    _column_warning_cache[cache_key] = now

    while len(_column_warning_cache) > _MAX_COLUMN_CACHE_SIZE:
        _column_warning_cache.popitem(last=False)

    return True


def _get_cached_table_columns(table_name: str, target: Any) -> list[str]:
    key = (table_name, _engine_cache_key(target))
    if key in _table_columns_cache:
        return list(_table_columns_cache[key])

    try:
        inspector = inspect(target)
        columns_info = inspector.get_columns(table_name)
        column_names = [col["name"] for col in columns_info]
    except Exception:
        table_meta = Base.metadata.tables.get(table_name)
        column_names = [c.name for c in table_meta.columns] if table_meta is not None else []

    _table_columns_cache[key] = list(column_names)
    return list(column_names)


class Universe(Base):
    """
    Schema definition for the universe table.

    In the production database, the universe table tracks symbol metadata,
    inclusion flags and timestamps.  We mirror those fields here so that
    ORM-generated upserts and queries align with the live schema.
    """

    __tablename__ = "universe"
    symbol = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    exchange = Column(String, nullable=True)
    market_cap = Column(Float, nullable=True)
    adv_usd_20 = Column(Float, nullable=True)
    included = Column(Boolean, nullable=False, default=True)
    last_updated = Column(DateTime, nullable=False)


class DailyBar(Base):
    """
    Schema definition for the daily_bars table.

    In the production database, daily_bars uses a composite primary key on
    (symbol, ts) and includes additional columns such as VWAP and trade_count.
    This definition mirrors the live schema to ensure ON CONFLICT upserts and
    column mappings work as expected.
    """

    __tablename__ = "daily_bars"
    # Composite primary key on (symbol, ts)
    symbol = Column(String, primary_key=True, nullable=False)
    ts = Column(Date, primary_key=True, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    # Additional columns from migration
    vwap = Column(Float, nullable=True)
    trade_count = Column(Integer, nullable=True)
    # Adjusted close is optional and may not be present in older schemas
    adj_close = Column(Float)


class Feature(Base):
    __tablename__ = "features"
    id = Column(Integer, primary_key=True)
    ts = Column(Date, index=True, nullable=False)
    symbol = Column(String, index=True, nullable=False)
    # price and momentum metrics
    ret_1d = Column(Float)
    ret_5d = Column(Float)
    ret_21d = Column(Float)
    mom_21 = Column(Float)
    mom_63 = Column(Float)
    vol_21 = Column(Float)
    vol_63 = Column(Float)
    turnover_21 = Column(Float)
    beta_63 = Column(Float)
    size_ln = Column(Float)
    # fundamental ratios
    f_pe_ttm = Column(Float)
    f_pb = Column(Float)
    f_ps_ttm = Column(Float)
    f_debt_to_equity = Column(Float)
    f_roa = Column(Float)
    f_roe = Column(Float)
    f_gross_margin = Column(Float)
    f_profit_margin = Column(Float)
    f_current_ratio = Column(Float)
    # macro features
    mkt_ret_1d = Column(Float)
    mkt_ret_5d = Column(Float)
    mkt_ret_21d = Column(Float)
    mkt_ret_63d = Column(Float)
    mkt_vol_21 = Column(Float)
    mkt_vol_63 = Column(Float)
    mkt_skew_21 = Column(Float)
    mkt_skew_63 = Column(Float)
    mkt_kurt_21 = Column(Float)
    mkt_kurt_63 = Column(Float)
    # cross-sectional z-scores
    cs_z_mom_21 = Column(Float)
    cs_z_mom_63 = Column(Float)
    cs_z_vol_21 = Column(Float)
    cs_z_vol_63 = Column(Float)
    cs_z_turnover_21 = Column(Float)
    cs_z_size_ln = Column(Float)
    cs_z_beta_63 = Column(Float)
    cs_z_f_pe_ttm = Column(Float)
    cs_z_f_pb = Column(Float)
    cs_z_f_ps_ttm = Column(Float)
    cs_z_f_debt_to_equity = Column(Float)
    cs_z_f_roa = Column(Float)
    cs_z_f_roe = Column(Float)
    cs_z_f_gross_margin = Column(Float)
    cs_z_f_profit_margin = Column(Float)
    cs_z_f_current_ratio = Column(Float)
    cs_z_mkt_ret_21d = Column(Float)
    cs_z_mkt_ret_63d = Column(Float)
    cs_z_mkt_vol_21 = Column(Float)
    cs_z_mkt_vol_63 = Column(Float)
    cs_z_mkt_skew_21 = Column(Float)
    cs_z_mkt_skew_63 = Column(Float)
    cs_z_mkt_kurt_21 = Column(Float)
    cs_z_mkt_kurt_63 = Column(Float)
    # optional sentiment/event columns
    signal_a = Column(Float)
    signal_a_lag1 = Column(Float)
    signal_b = Column(Float)
    signal_b_lag1 = Column(Float)
    __table_args__ = (Index("idx_features_symbol_ts", "symbol", "ts", unique=True),)


class Fundamentals(Base):
    """
    Schema definition for the fundamentals table.

    The production database does not include an auto-incrementing `id` column for
    fundamentals. Instead, the combination of (symbol, as_of) uniquely
    identifies each record.  We define both `symbol` and `as_of` as primary
    keys and drop the unused `id` column to match the live schema.  This
    prevents insertion errors caused by pandas attempting to insert a missing
    `id` column.
    """

    __tablename__ = "fundamentals"
    # composite primary key on (symbol, as_of)
    symbol = Column(String, primary_key=True, index=True, nullable=False)
    as_of = Column(Date, primary_key=True, index=True, nullable=False)
    available_at = Column(Date, nullable=True)
    debt_to_equity = Column(Float)
    return_on_assets = Column(Float)
    return_on_equity = Column(Float)
    gross_margins = Column(Float)
    profit_margins = Column(Float)
    current_ratio = Column(Float)
    __table_args__ = (
        Index("idx_fundamentals_symbol_asof", "symbol", "as_of", unique=True),
    )


class Prediction(Base):
    __tablename__ = "predictions"
    symbol = Column(String(20), primary_key=True, nullable=False)
    ts = Column(Date, primary_key=True, nullable=False, index=True)
    model_version = Column(String(32), primary_key=True, nullable=False, index=True)
    horizon = Column(Integer, primary_key=True, nullable=False, default=5)
    created_at = Column(DateTime, primary_key=True, nullable=False, server_default=func.now())
    y_pred = Column(Float, nullable=False)
    __table_args__ = tuple(_PREDICTION_INDEXES)


class BacktestEquity(Base):
    __tablename__ = "backtest_equity"
    id = Column(Integer, primary_key=True)
    ts = Column(Date, index=True, nullable=False)
    equity = Column(Float)


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True)
    trade_date = Column(Date, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float)
    status = Column(String)
    broker_order_id = Column(String)
    client_order_id = Column(String)
    ts = Column(Date, default=date.today)


class Position(Base):
    __tablename__ = "current_positions"
    id = Column(Integer, primary_key=True)
    symbol = Column(String, unique=True, nullable=False)
    shares = Column(Float, nullable=False)
    cost_basis = Column(Float)
    ts = Column(Date, default=date.today)


class TargetPosition(Base):
    __tablename__ = "target_positions"
    id = Column(Integer, primary_key=True)
    ts = Column(Date, index=True, nullable=False)
    symbol = Column(String, index=True, nullable=False)
    weight = Column(Float, nullable=False)
    price = Column(Float)
    target_shares = Column(Float, nullable=False)
    __table_args__ = (
        Index("idx_target_positions_ts_symbol", "ts", "symbol", unique=True),
    )


def create_tables() -> None:
    """Create all defined tables; add new columns if needed."""
    Base.metadata.create_all(bind=engine)


def _max_bind_params_for_connection(connection: Any) -> int:
    """Return the maximum number of bind parameters supported by a connection."""

    dialect_name: Optional[str] = None

    try:
        dialect = getattr(connection, "dialect", None)
        if dialect and getattr(dialect, "name", None):
            dialect_name = dialect.name
    except Exception:  # pragma: no cover - defensive for mocks
        dialect_name = None

    if not dialect_name:
        engine_obj = getattr(connection, "engine", None)
        if engine_obj is not None:
            try:
                dialect_name = engine_obj.dialect.name
            except Exception:  # pragma: no cover - defensive for mocks
                url = getattr(engine_obj, "url", None)
                if url is not None:
                    dialect_name = str(url)

    if not dialect_name and engine is not None:
        try:
            dialect_name = engine.dialect.name
        except Exception:  # pragma: no cover - extremely defensive
            dialect_name = None

    normalized = str(dialect_name or "").lower()

    if "postgres" in normalized or "psycopg" in normalized:
        return 65535
    if "mysql" in normalized or "mariadb" in normalized:
        return 65535
    if "oracle" in normalized:
        return 65535
    if "bigquery" in normalized:
        return 100000
    if "sqlite" in normalized:
        return 999

    # Fallback to SQLite's conservative default when dialect is unknown.
    return 999


def upsert_dataframe(
    df: pd.DataFrame,
    table: Any,
    conflict_cols: Optional[list[str]] = None,
    update_cols: Optional[list[str]] = None,
    chunk_size: Optional[int] = None,
    conn: Optional[Connection] = None,
) -> None:
    """
    Append a DataFrame to the given table.  Unknown columns are dropped and missing
    columns are filled with None.  Accepts conflict_cols and update_cols for
    backward compatibility.  Uses chunked inserts to avoid Postgres parameter
    limits.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to insert
    table : Any
        Table object or name
    conflict_cols : Optional[list[str]]
        Columns used to identify conflicting records.  When omitted the function
        falls back to the table's primary key definition.
    update_cols : Optional[list[str]]
        Explicit list of columns to update on conflict.  Defaults to all columns
        except the conflict columns when not provided.
    chunk_size : Optional[int]
        Desired number of rows per insert batch.  The function automatically
        caps this value based on the backend parameter limit.
    conn : Optional[Connection]
        Optional SQLAlchemy connection to reuse.  When not supplied the global
        engine is used and the function manages its own transaction.
    """
    if df is None or df.empty:
        return
    # determine table_name from object or string
    if hasattr(table, "__tablename__"):
        table_name = table.__tablename__
    elif hasattr(table, "name") and not isinstance(table, type):
        table_name = table.name
    else:
        table_name = str(table)

    if table_name in Base.metadata.tables:
        table_obj = Base.metadata.tables[table_name]
    elif hasattr(table, "__table__"):
        table_obj = table.__table__
    else:
        raise ValueError(f"Unknown table: {table_name}")

    target_engine = conn.engine if conn is not None else engine

    # gather valid columns for the target table using live inspection of the database
    valid_cols = _get_cached_table_columns(table_name, target_engine)
    if not valid_cols:
        valid_cols = [c.name for c in table_obj.columns]

    if not valid_cols:
        raise ValueError(f"No columns available for table: {table_name}")

    dropped_cols = {c for c in df.columns if c not in valid_cols}
    if dropped_cols and _should_log_column_warning(table_name, dropped_cols):
        if dropped_cols == {"adj_close"}:
            logger.debug("Dropping adj_close column not present in %s", table_name)
        else:
            logger.warning(
                "Dropping columns not present in %s: %s",
                table_name,
                ", ".join(sorted(dropped_cols)),
            )

    # retain only columns that exist in the table
    filtered_df = df.copy()[[c for c in df.columns if c in valid_cols]]
    # add missing columns with None so that pandas.to_sql includes them
    for col in valid_cols:
        if col not in filtered_df.columns:
            filtered_df[col] = None

    # Reorder columns to match the table definition for stable INSERT statements
    column_order = [col for col in valid_cols if col in filtered_df.columns]
    filtered_df = filtered_df[column_order]

    # Cast DataFrame columns to appropriate types based on the database schema.
    inspector = None

    try:
        inspector = inspect(target_engine)
        insp_cols = inspector.get_columns(table_name)
        float_cols = []
        int_cols = []
        date_cols = []
        datetime_cols = []
        bool_cols = []
        for col_info in insp_cols:
            col_name = col_info["name"]
            col_type = col_info["type"]
            if isinstance(col_type, Float):
                float_cols.append(col_name)
            elif isinstance(col_type, Integer):
                int_cols.append(col_name)
            elif isinstance(col_type, Date):
                date_cols.append(col_name)
            elif isinstance(col_type, DateTime):
                datetime_cols.append(col_name)
            elif isinstance(col_type, Boolean):
                bool_cols.append(col_name)
        for col in float_cols:
            if col in filtered_df.columns:
                filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce")
        for col in int_cols:
            if col in filtered_df.columns:
                filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce").astype("Int64")
        for col in date_cols:
            if col in filtered_df.columns:
                filtered_df[col] = pd.to_datetime(filtered_df[col], errors="coerce").dt.date
        for col in datetime_cols:
            if col in filtered_df.columns:
                filtered_df[col] = pd.to_datetime(filtered_df[col], errors="coerce")
        for col in bool_cols:
            if col in filtered_df.columns:
                filtered_df[col] = filtered_df[col].astype(bool)
    except Exception:
        # If inspection fails, silently skip casting; invalid types will raise DB errors
        pass

    # prepare metadata objects used for conflict resolution
    try:
        if inspector is None:
            inspector = inspect(target_engine)
        pk_columns = inspector.get_pk_constraint(table_name).get("constrained_columns", [])
    except Exception:
        pk_columns = [col.name for col in table_obj.primary_key.columns]

    dedupe_cols = pk_columns or (conflict_cols or [])
    if dedupe_cols:
        filtered_df = filtered_df.drop_duplicates(subset=dedupe_cols, keep="last")

    # Determine update columns for parameter counting
    if conflict_cols is None:
        conflict_cols = pk_columns
    effective_conflict_cols = conflict_cols or []

    if update_cols is None:
        update_cols = [c for c in filtered_df.columns if c not in effective_conflict_cols]

    params_per_row = len(filtered_df.columns)
    if effective_conflict_cols:
        params_per_row += len(update_cols)

    safety_margin = max(10, params_per_row // 10) if params_per_row else 0

    # Acquire connection / transaction handling
    connection = conn if conn is not None else target_engine.connect()
    close_connection = conn is None
    transaction = None

    try:
        max_bind_params = _max_bind_params_for_connection(connection)
    except Exception:
        max_bind_params = 999

    effective_limit = max_bind_params - safety_margin
    if effective_limit <= 0 or params_per_row == 0:
        calculated_chunksize = 1
    else:
        calculated_chunksize = max(1, effective_limit // max(1, params_per_row))

    if chunk_size is not None:
        chunksize = max(1, min(chunk_size, calculated_chunksize))
    else:
        chunksize = max(1, min(1000, calculated_chunksize))

    def _iter_chunks(seq: list[Any], size: int) -> Iterator[list[Any]]:
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    try:
        if not connection.in_transaction():
            transaction = connection.begin()

        if effective_conflict_cols and not filtered_df.empty:
            pk_cols = [table_obj.c[col] for col in effective_conflict_cols if col in table_obj.c]
            if pk_cols:
                pk_tuples = list(filtered_df[effective_conflict_cols].itertuples(index=False, name=None))
                for chunk_values in _iter_chunks(pk_tuples, 1000):
                    delete_stmt = table_obj.delete().where(tuple_(*pk_cols).in_(chunk_values))
                    try:
                        connection.execute(delete_stmt)
                    except ProgrammingError as exc:
                        logger.warning(
                            "Failed to delete existing rows for %s due to %s; continuing with insert",
                            table_name,
                            exc,
                        )

        if not filtered_df.empty:
            filtered_df.to_sql(
                table_name,
                con=connection,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=chunksize,
            )

        if transaction is not None:
            transaction.commit()
    except Exception:
        if transaction is not None:
            transaction.rollback()
        raise
    finally:
        if close_connection:
            connection.close()

