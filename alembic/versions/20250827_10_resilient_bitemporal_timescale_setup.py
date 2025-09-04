"""Resilient bitemporal + optional TimescaleDB setup.

Revision ID: 20250827_10
Revises: 20250825_07
Create Date: 2025-08-27
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect, text

revision = "20250827_10"
down_revision = "20250825_07"
branch_labels = None
depends_on = None


def _has_column(insp, table: str, column: str) -> bool:
    try:
        cols = [c["name"] for c in insp.get_columns(table)]
        return column in cols
    except Exception:
        return False


def _safe_add_column(table: str, column: sa.Column) -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    if not _has_column(insp, table, column.name):
        try:
            with op.batch_alter_table(table) as batch_op:
                batch_op.add_column(column)
        except Exception:
            pass


def _safe_exec(sql: str) -> None:
    try:
        op.execute(text(sql))
    except Exception:
        pass


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)

    _safe_add_column(
        "fundamentals",
        sa.Column("knowledge_date", sa.Date(), nullable=True),
    )

    try:
        insp.get_columns("shares_outstanding")
        _safe_add_column(
            "shares_outstanding",
            sa.Column("knowledge_date", sa.Date(), nullable=True),
        )
    except Exception:
        pass

    _safe_exec(
        "CREATE INDEX IF NOT EXISTS ix_fundamentals_bitemporal ON fundamentals (symbol, as_of, knowledge_date)"
    )
    _safe_exec(
        "CREATE INDEX IF NOT EXISTS ix_shares_outstanding_bitemporal ON shares_outstanding (symbol, as_of, knowledge_date)"
    )

    if bind.dialect.name.lower().startswith("postgres"):
        _safe_exec(
            """
DO $$
BEGIN
  BEGIN
    CREATE EXTENSION IF NOT EXISTS timescaledb;
  EXCEPTION WHEN others THEN
    RAISE NOTICE 'TimescaleDB extension not created – skipping';
  END;

  IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
    BEGIN
      PERFORM create_hypertable('daily_bars', 'ts', if_not_exists => TRUE);
    EXCEPTION WHEN others THEN
      RAISE NOTICE 'Hypertable daily_bars skipped: %', SQLERRM;
    END;

    BEGIN
      PERFORM create_hypertable('features', 'ts', if_not_exists => TRUE);
    EXCEPTION WHEN others THEN
      RAISE NOTICE 'Hypertable features skipped: %', SQLERRM;
    END;

    BEGIN
      PERFORM create_hypertable('predictions', 'ts', if_not_exists => TRUE);
    EXCEPTION WHEN others THEN
      RAISE NOTICE 'Hypertable predictions skipped: %', SQLERRM;
    END;

    BEGIN
      PERFORM create_hypertable('alt_signals', 'ts', if_not_exists => TRUE);
    EXCEPTION WHEN others THEN
      RAISE NOTICE 'Hypertable alt_signals skipped: %', SQLERRM;
    END;

    BEGIN
      PERFORM create_hypertable('short_borrow', 'ts', if_not_exists => TRUE);
    EXCEPTION WHEN others THEN
      RAISE NOTICE 'Hypertable short_borrow skipped: %', SQLERRM;
    END;

    BEGIN
      PERFORM create_hypertable('backtest_equity', 'ts', if_not_exists => TRUE);
    EXCEPTION WHEN others THEN
      RAISE NOTICE 'Hypertable backtest_equity skipped: %', SQLERRM;
    END;

    BEGIN
      PERFORM create_hypertable('universe_history', 'as_of', if_not_exists => TRUE);
    EXCEPTION WHEN others THEN
      RAISE NOTICE 'Hypertable universe_history skipped: %', SQLERRM;
    END;

    BEGIN
      PERFORM create_hypertable('option_overlays', 'as_of', if_not_exists => TRUE);
    EXCEPTION WHEN others THEN
      RAISE NOTICE 'Hypertable option_overlays skipped: %', SQLERRM;
    END;

    BEGIN
      PERFORM create_hypertable('trades', 'ts', if_not_exists => TRUE);
    EXCEPTION WHEN others THEN
      RAISE NOTICE 'Hypertable trades skipped: %', SQLERRM;
    END;
  ELSE
    RAISE NOTICE 'TimescaleDB extension not installed – skipping hypertable configuration';
  END IF;
END $$;
"""
        )


def downgrade() -> None:
    try:
        op.execute(text("DROP INDEX IF EXISTS ix_shares_outstanding_bitemporal"))
    except Exception:
        pass
    try:
        op.execute(text("DROP INDEX IF EXISTS ix_fundamentals_bitemporal"))
    except Exception:
        pass

    try:
        with op.batch_alter_table("fundamentals") as batch_op:
            batch_op.drop_column("knowledge_date")
    except Exception:
        pass
    try:
        insp = inspect(op.get_bind())
        insp.get_columns("shares_outstanding")
        with op.batch_alter_table("shares_outstanding") as batch_op:
            batch_op.drop_column("knowledge_date")
    except Exception:
        pass
