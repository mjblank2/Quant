from __future__ import annotations

import os
from datetime import date
from typing import Any, Optional

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
)
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
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    dsn = _normalise_dsn(dsn)
    return create_engine(dsn, pool_pre_ping=True)


engine = _create_engine_from_env()
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


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
    y_pred = Column(Float, nullable=False)
    horizon = Column(Integer, nullable=False, default=5)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    __table_args__ = (
        Index("ix_predictions_ts", "ts"),
        Index("ix_predictions_ts_model", "ts", "model_version"),
    )


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


def upsert_dataframe(
    df: pd.DataFrame,
    table: Any,
    conflict_cols: Optional[list[str]] = None,
    update_cols: Optional[list[str]] = None,
    chunk_size: Optional[int] = None,
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
        Conflict columns (for backward compatibility, not used in current implementation)
    update_cols : Optional[list[str]]
        Update columns (for backward compatibility, not used in current implementation)
    chunk_size : Optional[int]
        Number of rows to insert per batch. Defaults to 1000 if not specified.
    """
    if df is None or df.empty:
        return
    # determine table_name from object or string
    if hasattr(table, "name"):
        table_name = table.name
    elif hasattr(table, "__tablename__"):
        table_name = table.__tablename__
    else:
        table_name = str(table)

    if table_name not in Base.metadata.tables:
        raise ValueError(f"Unknown table: {table_name}")

    # gather valid columns for the target table using live inspection of the database
    valid_cols = set(c.name for c in Base.metadata.tables[table_name].columns)
    try:
        insp = inspect(engine)
        columns_info = insp.get_columns(table_name)
        valid_cols = {col["name"] for col in columns_info}
    except Exception:
        # fallback to metadata-defined columns if inspection fails
        valid_cols = {c.name for c in Base.metadata.tables[table_name].columns}

    # retain only columns that exist in the table
    filtered_df = df.copy()[[c for c in df.columns if c in valid_cols]]
    # add missing columns with None so that pandas.to_sql includes them
    for col in valid_cols:
        if col not in filtered_df.columns:
            filtered_df[col] = None

    # Convert DataFrame columns to numeric for columns that are Float in the database.
    # When DataFrame dtypes are object (e.g. strings), pandas will generate VARCHAR
    # parameters in to_sql, causing a datatype mismatch.  Inspect the table schema
    # and coerce any Float columns to numeric before insertion.
    try:
        insp_cols = inspect(engine).get_columns(table_name)
        float_cols = [col_info["name"] for col_info in insp_cols if isinstance(col_info["type"], Float)]
        for col in float_cols:
            if col in filtered_df.columns:
                filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce")
    except Exception:
        # If inspection fails, silently skip casting; invalid types will raise DB errors
        pass

    # choose a sensible chunksize to avoid hitting Postgres parameter limits (65535)
    chunksize = chunk_size if chunk_size is not None else 1000

    filtered_df.to_sql(
        table_name,
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=chunksize,
    )
