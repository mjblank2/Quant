"""Backfill neutral defaults for new ML feature columns."""
from __future__ import annotations
from alembic import op
import sqlalchemy as sa

revision = "20250910_04"
down_revision = "20250910_03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Use individual UPDATEs guarded by column existence
    # Rationale: simple, explicit; avoids overwriting any pre-populated values.
    from sqlalchemy import inspect
    bind = op.get_bind()
    inspector = inspect(bind)

    # Helper to check column existence
    def col_exists(col: str) -> bool:
        existing = {c['name'] for c in inspector.get_columns('features')}
        return col in existing

    # Columns with neutral zero default
    zero_cols = ['reversal_5d_z', 'ivol_63', 'pead_event', 'pead_surprise_eps', 'pead_surprise_rev', 'russell_inout']
    for col in zero_cols:
        if col_exists(col):
            bind.execute(sa.text(f"UPDATE features SET {col}=0 WHERE {col} IS NULL"))

    # Leave forward-looking returns NULL (only fill if explicitly desired)
    # fwd_ret, fwd_ret_resid intentionally untouched; they will be populated by the feature pipeline.

    # Do NOT touch beta_63 / overnight_gap / illiq_21; they were legacy or already populated.


def downgrade() -> None:
    # Backfill is data-only; nothing to undo safely without risking user data.
    pass
