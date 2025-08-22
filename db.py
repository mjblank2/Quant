from __future__ import annotations
import os
from sqlalchemy import create_engine, String, Date, DateTime, Integer, Float, Boolean, BigInteger, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime, date
import pandas as pd
from config import DATABASE_URL

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        url = DATABASE_URL or os.getenv("DATABASE_URL") or os.getenv("DB_URL")
        if not url:
            raise RuntimeError("DATABASE_URL is required")
        _engine = create_engine(url, pool_pre_ping=True, future=True)
    return _engine

def SessionLocal():
    # Returns a Session instance for use as: with SessionLocal() as s:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False, future=True)()

class Base(DeclarativeBase):
    pass

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
    market_cap: Mapped[float | None] = mapped_column(Float, nullable=True)
    adv_usd_20: Mapped[float | None] = mapped_column(Float, nullable=True)
    included: Mapped[bool] = mapped_column(Boolean, default=True)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

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
    __table_args__ = (Index("ix_fundamentals_symbol_asof", "symbol", "as_of"),)

class AltSignal(Base):
    __tablename__ = "alt_signals"
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    ts: Mapped[date] = mapped_column(Date, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    __table_args__ = (Index("ix_alt_symbol_ts", "symbol", "ts"),)

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

class Position(Base):
    __tablename__ = "positions"
    ts: Mapped[date] = mapped_column(Date, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    weight: Mapped[float] = mapped_column(Float)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    shares: Mapped[int | None] = mapped_column(Integer, nullable=True)
    __table_args__ = (Index("ix_positions_ts_symbol", "ts", "symbol"),)

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
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    __table_args__ = (Index("ix_trades_status_id", "status", "id"),)

class BacktestEquity(Base):
    __tablename__ = "backtest_equity"
    ts: Mapped[date] = mapped_column(Date, primary_key=True)
    equity: Mapped[float] = mapped_column(Float)
    daily_return: Mapped[float] = mapped_column(Float)
    drawdown: Mapped[float] = mapped_column(Float)

def create_tables():
    Base.metadata.create_all(get_engine())

def upsert_dataframe(df: pd.DataFrame, table, conflict_cols: list[str], chunk_size: int = 50000):
    if df is None or df.empty:
        return
    from sqlalchemy.dialects.postgresql import insert
    for start in range(0, len(df), chunk_size):
        part = df.iloc[start:start+chunk_size]
        cols = list(part.columns)
        stmt = insert(table).values(part.to_dict(orient="records"))
        update_cols = {c: getattr(stmt.excluded, c) for c in cols if c not in conflict_cols}
        stmt = stmt.on_conflict_do_update(index_elements=conflict_cols, set_=update_cols)
        with get_engine().begin() as conn:
            conn.execute(stmt)

