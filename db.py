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
from sqlalchemy.orm import sessionmaker  # <— uncommented import

# ---------------------------------------------------------------------------
# Engine and session
# ---------------------------------------------------------------------------

def _normalise_dsn(url: str) -> str:
    """Normalise PostgreSQL DSNs to include the psycopg dialect."""
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

engine = _create_engine_from_env()
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ---------------------------------------------------------------------------
# ORM model definitions
# ---------------------------------------------------------------------------

class Universe(Base):
    __tablename__ = "universe"
    symbol = Column(String, primary_key=True)
    sector = Column(String, nullable=True)
    industry = Column(String, nullable=True)
    ts = Column(Date, index=True)

class DailyBar(Base):
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

    # cross-sectional z‑scores
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

    # sentiment/event signals and 1‑day lag (placeholders)
    signal_a = Column(Float)
    signal_a_lag1 = Column(Float)
    signal_b = Column(Float)
    signal_b_lag1 = Column(Float)

    __table_args__ = (
        Index("idx_features_symbol_ts", "symbol", "ts", unique=True),
    )

class Prediction(Base):
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
    __tablename__ = "backtest_equity"
    id = Column(Integer, primary_key=True)
    ts = Column(Date, index=True, nullable=False)
    equity = Column(Float)

class Trade(Base):
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
    """Create all tables defined in this module."""
    Base.metadata.create_all(bind=engine)

def upsert_dataframe(
    df: pd.DataFrame,
    table: Any,
    conflict_cols: Optional[list[str]] = None,
    update_cols: Optional[list[str]] = None,
) -> None:
    """
    Insert or update a pandas DataFrame into the specified table.

    Accepts ``conflict_cols`` and ``update_cols`` for backward compatibility,
    but ignores them since this helper always performs an append operation.
    Drops any unknown columns and adds missing columns as NULL.
    """
    if df is None or df.empty:
        return

    # Derive the table name (string) from class or Table object
    if hasattr(table, "name"):
        table_name = table.name
    elif hasattr(table, "__tablename__"):
        table_name = table.__tablename__
    else:
        table_name = str(table)

    if table_name not in Base.metadata.tables:
        raise ValueError(f"Unknown table: {table_name}")

    valid_cols = set(c.name for c in Base.metadata.tables[table_name].columns)
    filtered_df = df.copy()[[c for c in df.columns if c in valid_cols]]

    for col in valid_cols:
        if col not in filtered_df.columns:
            filtered_df[col] = None

    filtered_df.to_sql(
        table_name,
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
    )
