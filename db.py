"""
Database models and helper functions for the Quant trading system.

This module defines SQLAlchemy ORM models for all tables used by the
Quant research and trading pipeline, including the features table.  It
also provides functions to initialise the database and upsert pandas
DataFrames into the appropriate tables.

To use this module, set the `DATABASE_URL` environment variable to a
valid PostgreSQL connection string.  The connection string should use
`postgresql+psycopg://` as the scheme and include credentials if
necessary.  See SQLAlchemy documentation for more details.

Example usage:

    from db import Base, engine, create_tables, upsert_dataframe

    create_tables()  # creates tables if they do not exist
    # write a DataFrame of feature rows
    upsert_dataframe(features_df, Feature.__tablename__)
"""

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
    create_engine,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
    # from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------------------------
# Engine and session
# ---------------------------------------------------------------------------

# Normalise PostgreSQL connection strings to include the psycopg dialect.
def _normalise_dsn(url: str) -> str:
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://") and "+psycopg" not in url:
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _create_engine_from_env() -> Any:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "DATABASE_URL environment variable must be set to connect to the database"
        )
    dsn = _normalise_dsn(dsn)
    return create_engine(dsn, pool_pre_ping=True)


# Create global SQLAlchemy objects.  The engine is created lazily to avoid
# connecting to the database at import time in environments without DB access.
engine = _create_engine_from_env()
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ---------------------------------------------------------------------------
# ORM model definitions
# ---------------------------------------------------------------------------

class Universe(Base):
    """Universe of tradable symbols and associated metadata."""

    __tablename__ = "universe"

    symbol = Column(String, primary_key=True)
    sector = Column(String, nullable=True)
    industry = Column(String, nullable=True)
    ts = Column(Date, index=True)


class DailyBar(Base):
    """Daily OHLCV bars for each symbol."""

    __tablename__ = "daily_bars"

    id = Column(Integer, primary_key=True)
    ts = Column(Date, index=True, nullable=False)
    symbol = Column(String, index=True, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    adj_close = Column(Float)


class Feature(Base):
    """
    Model representing engineered features.  This table contains one row per
    symbol and date, with columns for momentum, volatility, market
    statistics, cross‑sectional z‑scores, macro features and fundamental
    ratios.  Update this class whenever you add new feature columns to the
    pipeline.
    """

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

    # market macro features (short and long horizons)
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

    # cross‑sectional z‑scores (add as needed)
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

    # optional sentiment/event signals and lags (add your own signals here)
    signal_a = Column(Float)
    signal_a_lag1 = Column(Float)
    signal_b = Column(Float)
    signal_b_lag1 = Column(Float)
    # Add more signal and signal_lag1 fields as needed.

    __table_args__ = (
        Index("idx_features_symbol_ts", "symbol", "ts", unique=True),
    )


class Prediction(Base):
    """Model predictions for each symbol and model version."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    ts = Column(Date, index=True, nullable=False)
    symbol = Column(String, index=True, nullable=False)
    model_version = Column(String, index=True, nullable=False)
    y_pred = Column(Float)

    __table_args__ = (
        Index("idx_predictions_symbol_ts", "symbol", "ts", "model_version", unique=True),
    )


class BacktestEquity(Base):
    """Equity curve for backtests."""

    __tablename__ = "backtest_equity"

    id = Column(Integer, primary_key=True)
    ts = Column(Date, index=True, nullable=False)
    equity = Column(Float)


class Trade(Base):
    """Executed or proposed trades."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    trade_date = Column(Date, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)  # 'buy' or 'sell'
    quantity = Column(Float, nullable=False)
    price = Column(Float)
    status = Column(String)
    broker_order_id = Column(String)
    client_order_id = Column(String)
    ts = Column(Date, default=date.today)


class Position(Base):
    """Current positions for each symbol."""

    __tablename__ = "current_positions"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, unique=True, nullable=False)
    shares = Column(Float, nullable=False)
    cost_basis = Column(Float)
    ts = Column(Date, default=date.today)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def create_tables() -> None:
    """
    Create all tables defined in this module.  If tables already exist,
    missing columns will be added automatically by SQLAlchemy's metadata.
    Call this at application start to ensure the database schema is
    up-to-date.
    """
    Base.metadata.create_all(bind=engine)

def upsert_dataframe(
    df: pd.DataFrame,
    table: Any,
    conflict_cols: Optional[list[str]] = None,
    update_cols: Optional[list[str]] = None,
) -> None:
    """
    Insert or update a pandas DataFrame into the specified table.

    This helper accepts either a table name (string) or a SQLAlchemy Table
    object.  It also accepts ``conflict_cols`` and ``update_cols`` parameters
    for backwards compatibility with older ingestion code.  These parameters
    are currently ignored because this implementation always performs a
    simple append operation.  Unknown columns in the input DataFrame will be
    dropped; missing columns will be added with NULL values.

    Args:
        df: DataFrame containing rows to insert/update.
        table: Name of the database table (str) or Table object.
        conflict_cols: Optional list of columns to use for conflict
            resolution when upserting.  Ignored in this implementation.
        update_cols: Optional list of columns to update on conflict.
            Ignored in this implementation.
    """
    # Do nothing if DataFrame is empty
    if df is None or df.empty:
        return

    # Derive the table name from either a string or a SQLAlchemy Table object
    if hasattr(table, "name"):
        table_name = table.name
    elif hasattr(table, "__tablename__"):
        table_name = table.__tablename__  # type: ignore
    else:
        table_name = str(table)

    # Check the table exists in metadata
    if table_name not in Base.metadata.tables:
        raise ValueError(f"Unknown table: {table_name}")

    # Determine valid columns for the target table
    valid_cols = set(c.name for c in Base.metadata.tables[table_name].columns)

    # Keep only valid columns from the DataFrame
    filtered_df = df.copy()[[c for c in df.columns if c in valid_cols]]

    # Add any missing columns with None (NULL) values so that the DataFrame
    # aligns with the database schema
    for col in valid_cols:
        if col not in filtered_df.columns:
            filtered_df[col] = None

    # Perform the insert; SQLAlchemy will handle the rest.  Setting
    # ``if_exists='append'`` will append rows.  We use the multi insert
    # method to improve performance.
    filtered_df.to_sql(
        table_name,
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
    )
