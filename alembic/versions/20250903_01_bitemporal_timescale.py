"""Resilient bitemporal + optional TimescaleDB setup.

- Guarantees fundamentals.available_at exists before referencing it
- Adds fundamentals.knowledge_date and backfills it from available_at (or as_of + 1 day)
- Ensures helpful indexes (symbol, available_at) and (symbol, as_of, knowledge_date)
- Creates shares_outstanding (if missing) with knowledge_date and indexes
- Optionally converts daily_bars to a TimescaleDB hypertable (PostgreSQL only)

Safe to re-run; all DDL guarded by inspector checks or IF NOT EXISTS.

Revision ID: 20250903_01
Revises: 20250827_12
Create Date: 2025-09-03
"""
from __future__ import annotations

import os
from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine import Connection

revision = "20250903_01"
down_revision = "20250827_12"
branch_labels = None
depends_on = None


def _tables(conn: Connection) -> set[str]:
    insp = sa.inspect(conn)
    return set(insp.get_table_names())


def _columns(conn: Connection, table: str) -> set[str]:
    insp = sa.inspect(conn)
    try:
        return {c["name"] for c in insp.get_columns(table)}
    except Exception:
        return set()


def _indexes(conn: Connection, table: str) -> set[str]:
    insp = sa.inspect(conn)
    try:
        return {i.get("name") for i in insp.get_indexes(table)}
    except Exception:
        return set()


def _ensure_fundamentals_schema(conn: Connection) -> None:
    # Create fundamentals if completely missing (defensive; normally exists from 20250822_02)
    if "fundamentals" not in _tables(conn):
        op.create_table(
            "fundamentals",
            sa.Column("symbol", sa.String(20), primary_key=True),
            sa.Column("as_of", sa.Date(), primary_key=True),
            sa.Column("pe_ttm", sa.Float(), nullable=True),
            sa.Column("pb", sa.Float(), nullable=True),
            sa.Column("ps_ttm", sa.Float(), nullable=True),
            sa.Column("debt_to_equity", sa.Float(), nullable=True),
            sa.Column("return_on_assets", sa.Float(), nullable=True),
            sa.Column("gross_margins", sa.Float(), nullable=True),
            sa.Column("profit_margins", sa.Float(), nullable=True),
            sa.Column("current_ratio", sa.Float(), nullable=True),
        )
        try:
            op.create_index("ix_fundamentals_symbol_asof", "fundamentals", ["symbol", "as_of"])
        except Exception:
            pass

    cols = _columns(conn, "fundamentals")

    # Add available_at if missing
    if "available_at" not in cols:
        try:
            with op.batch_alter_table("fundamentals") as batch:
                batch.add_column(sa.Column("available_at", sa.Date(), nullable=True))
        except Exception:
            pass

    # Add knowledge_date if missing
    cols = _columns(conn, "fundamentals")
    if "knowledge_date" not in cols:
        try:
            with op.batch_alter_table("fundamentals") as batch:
                batch.add_column(sa.Column("knowledge_date", sa.Date(), nullable=True))
        except Exception:
            pass

    # Indexes
    idxs = _indexes(conn, "fundamentals")
    if "ix_fundamentals_symbol_available_at" not in idxs and "available_at" in _columns(conn, "fundamentals"):
        try:
            op.create_index(
                "ix_fundamentals_symbol_available_at",
                "fundamentals",
                ["symbol", "available_at"],
                unique=False,
            )
        except Exception:
            pass

    if "ix_fundamentals_bitemporal" not in idxs and {"as_of", "knowledge_date"}.issubset(_columns(conn, "fundamentals")):
        try:
            op.create_index(
                "ix_fundamentals_bitemporal",
                "fundamentals",
                ["symbol", "as_of", "knowledge_date"],
                unique=False,
            )
        except Exception:
            pass


def _backfill_fundamentals_knowledge_date(conn: Connection) -> None:
    # Only attempt if table/columns exist
    cols = _columns(conn, "fundamentals")
    if not {"as_of", "knowledge_date"}.issubset(cols):
        return

    # Build COALESCE safely: if available_at exists, use it; else fallback to as_of + 1 day
    uses_available = "available_at" in cols
    if conn.dialect.name == "postgresql":
        if uses_available:
            sql = sa.text(
                """
                UPDATE fundamentals
                SET knowledge_date = COALESCE(knowledge_date, COALESCE(available_at, as_of + INTERVAL '1 day'))
                WHERE knowledge_date IS NULL
                """
            )
        else:
            sql = sa.text(
                """
                UPDATE fundamentals
                SET knowledge_date = COALESCE(knowledge_date, as_of + INTERVAL '1 day')
                WHERE knowledge_date IS NULL
                """
            )
    else:
        # SQLite syntax
        if uses_available:
            sql = sa.text(
                """
                UPDATE fundamentals
                SET knowledge_date = COALESCE(knowledge_date, COALESCE(available_at, DATE(as_of, '+1 day')))
                WHERE knowledge_date IS NULL
                """
            )
        else:
            sql = sa.text(
                """
                UPDATE fundamentals
                SET knowledge_date = COALESCE(knowledge_date, DATE(as_of, '+1 day'))
                WHERE knowledge_date IS NULL
                """
            )
    try:
        conn.execute(sql)
    except Exception:
        # Soft-fail; better to have nullable knowledge_date than block migration
        pass


def _ensure_shares_outstanding(conn: Connection) -> None:
    # Create table if missing (align with ORM)
    if "shares_outstanding" not in _tables(conn):
        op.create_table(
            "shares_outstanding",
            sa.Column("symbol", sa.String(20), primary_key=True),
            sa.Column("as_of", sa.Date(), primary_key=True),
            sa.Column("shares", sa.BigInteger(), nullable=False),
            sa.Column("source", sa.String(32), nullable=True),
            sa.Column("knowledge_date", sa.Date(), nullable=True),
        )
        try:
            op.create_index("ix_shares_symbol_asof", "shares_outstanding", ["symbol", "as_of"])
        except Exception:
            pass
        try:
            op.create_index(
                "ix_shares_outstanding_bitemporal",
                "shares_outstanding",
                ["symbol", "as_of", "knowledge_date"],
            )
        except Exception:
            pass
        return

    # If exists, ensure knowledge_date + indexes
    cols = _columns(conn, "shares_outstanding")
    if "knowledge_date" not in cols:
        try:
            with op.batch_alter_table("shares_outstanding") as batch:
                batch.add_column(sa.Column("knowledge_date", sa.Date(), nullable=True))
        except Exception:
            pass

    idxs = _indexes(conn, "shares_outstanding")
    if "ix_shares_symbol_asof" not in idxs and {"symbol", "as_of"}.issubset(_columns(conn, "shares_outstanding")):
        try:
            op.create_index("ix_shares_symbol_asof", "shares_outstanding", ["symbol", "as_of"])
        except Exception:
            pass

    if "ix_shares_outstanding_bitemporal" not in idxs and {"symbol", "as_of", "knowledge_date"}.issubset(
        _columns(conn, "shares_outstanding")
    ):
        try:
            op.create_index(
                "ix_shares_outstanding_bitemporal",
                "shares_outstanding",
                ["symbol", "as_of", "knowledge_date"],
            )
        except Exception:
            pass


def _maybe_enable_timescale(conn: Connection) -> None:
    # Only attempt on PostgreSQL
    if conn.dialect.name != "postgresql":
        return

    enable_ts = (os.getenv("ENABLE_TIMESCALEDB", "true") or "true").lower() in {"1", "true", "yes", "y"}
    if not enable_ts:
        return

    # Try to create extension; ignore if not installed
    try:
        conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
    except Exception:
        return  # timescaledb not available

    # Create hypertable for daily_bars(ts) if not already hypertable
    # Guard with catalog check; wrap to tolerate missing catalog on old PG
    try:
        conn.execute(
            sa.text(
                """
DO $$
BEGIN
  IF EXISTS (
       SELECT 1
       FROM information_schema.tables
       WHERE table_schema = current_schema()
         AND table_name = 'daily_bars'
     )
  THEN
    IF EXISTS (
         SELECT 1 FROM pg_class c
         JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = current_schema()
           AND c.relname = 'daily_bars'
       )
    THEN
      -- Only convert if not already a hypertable
      IF NOT EXISTS (
           SELECT 1
           FROM _timescaledb_catalog.hypertable ht
           JOIN pg_class c ON c.oid = ht.main_table_relid
           JOIN pg_namespace n ON n.oid = c.relnamespace
           WHERE n.nspname = current_schema()
             AND c.relname = 'daily_bars'
         )
      THEN
        PERFORM create_hypertable(format('%I.daily_bars', current_schema()), 'ts', if_not_exists => TRUE);
      END IF;
    END IF;
  END IF;
EXCEPTION WHEN undefined_table OR undefined_function THEN
  -- timescaledb catalog or function not available
  NULL;
END$$;
                """
            )
        )
    except Exception:
        pass


def upgrade() -> None:
    bind = op.get_bind()

    # 1) Fundamentals: ensure columns/indexes exist before any UPDATE referencing them
    _ensure_fundamentals_schema(bind)

    # 2) Backfill knowledge_date (safe COALESCE of available_at or as_of+1day)
    _backfill_fundamentals_knowledge_date(bind)

    # 3) Shares Outstanding: ensure table/columns/indexes (aligns with ORM expectations)
    _ensure_shares_outstanding(bind)

    # 4) Optional: TimescaleDB hypertable for daily_bars
    _maybe_enable_timescale(bind)


def downgrade() -> None:
    # Best-effort clean-up. We do NOT drop available_at on fundamentals because
    # it may have been added by earlier migrations (and is used by code).
    try:
        op.drop_index("ix_fundamentals_bitemporal", table_name="fundamentals")
    except Exception:
        pass
    try:
        op.drop_index("ix_fundamentals_symbol_available_at", table_name="fundamentals")
    except Exception:
        pass
    try:
        with op.batch_alter_table("fundamentals") as batch:
            batch.drop_column("knowledge_date")
    except Exception:
        pass

    # Shares outstanding indexes/column (keep table)
    try:
        op.drop_index("ix_shares_outstanding_bitemporal", table_name="shares_outstanding")
    except Exception:
        pass
    try:
        op.drop_index("ix_shares_symbol_asof", table_name="shares_outstanding")
    except Exception:
        pass
    try:
        with op.batch_alter_table("shares_outstanding") as batch:
            batch.drop_column("knowledge_date")
    except Exception:
        pass
