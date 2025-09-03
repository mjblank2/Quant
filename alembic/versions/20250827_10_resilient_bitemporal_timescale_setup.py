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
            # Ignore if permissions or other issues; keep migration progressing
            pass


def _safe_exec(sql: str) -> None:
    try:
        op.execute(text(sql))
    except Exception:
        # Ignore failures to keep migration resilient
        pass


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)

    # 1) Bitemporal columns (idempotent)
    # fundamentals: available_at (already present in prior migrations), knowledge_date, plus indexes
    _safe_add_column(
        "fundamentals",
        sa.Column("knowledge_date", sa.Date(), nullable=True),
    )

    # shares_outstanding table may not exist yet in some DBs; guard everything
    try:
        insp.get_columns("shares_outstanding")
        _safe_add_column(
            "shares_outstanding",
            sa.Column("knowledge_date", sa.Date(), nullable=True),
        )
    except Exception:
        pass

    # 2) Idempotent indexes for bitemporal usage
    # CREATE INDEX IF NOT EXISTS supported on Postgres
    _safe_exec(
        "CREATE INDEX IF NOT EXISTS ix_fundamentals_bitemporal ON fundamentals (symbol, as_of, knowledge_date)"
    )
    _safe_exec(
        "CREATE INDEX IF NOT EXISTS ix_shares_outstanding_bitemporal ON shares_outstanding (symbol, as_of, knowledge_date)"
    )

    # 3) Optional TimescaleDB setup (safe/no-op if unavailable or insufficient privileges)
    # - Only on Postgres
    if bind.dialect.name.lower().startswith("postgres"):
        # Try to create extension (ignore permission errors), then create hypertables if extension is present
        _safe_exec(
            """
DO $$
BEGIN
  -- Try to create the extension; permission errors will be caught
  BEGIN
    CREATE EXTENSION IF NOT EXISTS timescaledb;
  EXCEPTION WHEN others THEN
    RAISE NOTICE 'TimescaleDB extension not created (permission/plan?) – skipping';
  END;

  IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
    -- Convert key time-series tables to hypertables (if not already)
    -- Each call is wrapped to avoid aborting migration
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

    -- universe_history uses as_of for time
    BEGIN
      PERFORM create_hypertable('universe_history', 'as_of', if_not_exists => TRUE);
    EXCEPTION WHEN others THEN
      RAISE NOTICE 'Hypertable universe_history skipped: %', SQLERRM;
    END;

    -- option_overlays uses as_of for time
    BEGIN
      PERFORM create_hypertable('option_overlays', 'as_of', if_not_exists => TRUE);
    EXCEPTION WHEN others THEN
      RAISE NOTICE 'Hypertable option_overlays skipped: %', SQLERRM;
    END;

    -- trades has ts (timestamp); can be large – optional
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
    # Keep downgrade minimal and safe; do not attempt to drop extensions/hypertables automatically.
    # Optionally drop added indexes/columns if present.
    try:
        op.execute(
            text(
                "DROP INDEX IF EXISTS ix_shares_outstanding_bitemporal"
            )
        )
    except Exception:
        pass
    try:
        op.execute(
            text(
                "DROP INDEX IF EXISTS ix_fundamentals_bitemporal"
            )
        )
    except Exception:
        pass

    # Drop knowledge_date columns if they exist (safe in batch_alter)
    try:
        with op.batch_alter_table("fundamentals") as batch_op:
            batch_op.drop_column("knowledge_date")
    except Exception:
        pass
    try:
        insp = inspect(op.get_bind())
        insp.get_columns("shares_outstanding")  # ensure table exists
        with op.batch_alter_table("shares_outstanding") as batch_op:
            batch_op.drop_column("knowledge_date")
    except Exception:
        pass
