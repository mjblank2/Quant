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

    if "available_at" not in cols:
        try:
            with op.batch_alter_table("fundamentals") as batch:
                batch.add_column(sa.Column("available_at", sa.Date(), nullable=True))
        except Exception:
            pass

    cols = _columns(conn, "fundamentals")
    if "knowledge_date" not in cols:
        try:
            with op.batch_alter_table("fundamentals") as batch:
                batch.add_column(sa.Column("knowledge_date", sa.Date(), nullable=True))
        except Exception:
            pass

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
    cols = _columns(conn, "fundamentals")
    if not {"as_of", "knowledge_date"}.issubset(cols):
        return

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
        if conn.dialect.name == "postgresql":
            with conn.begin_nested():
                conn.execute(sql)
        else:
            conn.execute(sql)
    except Exception as e:
        print(f"Warning: knowledge_date backfill failed: {e}")
        pass


def _ensure_shares_outstanding(conn: Connection) -> None:
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
    if conn.dialect.name != "postgresql":
        return

    enable_ts = (os.getenv("ENABLE_TIMESCALEDB", "true") or "true").lower() in {"1", "true", "yes", "y"}
    if not enable_ts:
        return

    try:
        with conn.begin_nested():
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
    except Exception:
        return

    try:
        with conn.begin_nested():
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
  NULL;
END$$;
                    """
                )
            )
    except Exception as e:
        print(f"Warning: TimescaleDB hypertable creation failed: {e}")
        pass


def upgrade() -> None:
    bind = op.get_bind()
    _ensure_fundamentals_schema(bind)
    _backfill_fundamentals_knowledge_date(bind)
    _ensure_shares_outstanding(bind)
    _maybe_enable_timescale(bind)


def downgrade() -> None:
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
