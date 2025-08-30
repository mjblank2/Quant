from __future__ import annotations
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

# Allow tests to run without a configured database URL by falling back to an
# in-memory SQLite database. This avoids importing `db` from failing when the
# `DATABASE_URL` environment variable is unset, which is the case in the test
# environment for broker integration.
if not DATABASE_URL:
    log.warning("DATABASE_URL not set; using in-memory SQLite database")
    engine = create_engine("sqlite:///:memory:", future=True)
else:
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
    f_pe_ttm: Mapped[float | None] = mapped_column(Float, nullable=True)
    f_pb: Mapped[float | None] = mapped_column(Float, nullable=True)
    f_ps_ttm: Mapped[float | None] = mapped_column(Float, nullable=True)
    f_debt_to_equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    f_roa: Mapped[float | None] = mapped_column(Float, nullable=True)
    f_gm: Mapped[float | None] = mapped_column(Float, nullable=True)
    f_profit_margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    f_current_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    beta_63: Mapped[float | None] = mapped_column(Float, nullable=True)  # v16 extension
    overnight_gap: Mapped[float | None] = mapped_column(Float, nullable=True)  # v16 extension
    illiq_21: Mapped[float | None] = mapped_column(Float, nullable=True)  # v16 extension
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
    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True)
    data_date: Mapped[date] = mapped_column(Date, nullable=False)
    ingestion_timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    source_timestamp: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    lineage_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    __table_args__ = (
        Index("ix_lineage_table_symbol_date", "table_name", "symbol", "data_date"),
        Index("ix_lineage_ingestion_time", "ingestion_timestamp"),
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
from contextlib import nullcontext


def upsert_dataframe(df: pd.DataFrame, table, conflict_cols: list[str], chunk_size: int = 50000, conn=None):
    """
    Insert/update DataFrame rows into 'table' with ON CONFLICT handling.
    Automatically batches statements to stay under PostgreSQL's parameter limit.
    """
    if df is None or df.empty:
        return
    df = df.replace({pd.NA: None, np.nan: None})

    # Safety: PostgreSQL's parameter limit varies by configuration.
    # Use a very conservative limit to ensure compatibility across different PostgreSQL setups.
    # Some configurations may have limits as low as 16,000 parameters.
    MAX_BIND_PARAMS = 10000  # Very conservative to prevent parameter limit errors

    ctx = engine.begin() if conn is None else nullcontext(conn)
    with ctx as connection:
        cols_all = list(df.columns)
        # rows per statement bounded by MAX_BIND_PARAMS / num_columns
        # Add additional safety: ensure we don't exceed reasonable batch sizes
        theoretical_max_rows = MAX_BIND_PARAMS // max(1, len(cols_all))
        per_stmt_rows = max(1, min(chunk_size, theoretical_max_rows, 1000))  # Cap at 1000 rows max

        for start in range(0, len(df), per_stmt_rows):
            part = df.iloc[start:start + per_stmt_rows]
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
                # If we still hit parameter limits, retry with smaller batches
                # Also handle PostgreSQL transaction abort errors
                if (("parameter" in str(e).lower() or "InFailedSqlTransaction" in str(e)) and len(records) > 10):
                    log.warning(f"Parameter limit or transaction abort error with {len(records)} records, retrying with smaller batches")
                    # Rollback the current transaction to clear the aborted state
                    try:
                        connection.rollback()
                    except Exception:
                        pass  # Ignore rollback errors - transaction might already be rolled back

                    # Recursively call with much smaller chunks and no connection (will create new transaction)
                    smaller_df = pd.DataFrame(records)
                    upsert_dataframe(smaller_df, table, conflict_cols, chunk_size=10, conn=None)
                else:
                    # Re-raise if it's not a parameter limit issue or if we're already at minimum size
                    raise
