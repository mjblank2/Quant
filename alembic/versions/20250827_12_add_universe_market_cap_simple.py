"""Add market_cap column to universe table if not exists (simplified).

Revision ID: 20250827_12
Revises: 20250827_add_task_status_table
Create Date: 2025-08-27

"""

from alembic import op
import sqlalchemy as sa


revision = "20250827_12"
down_revision = "20250827_add_task_status_table"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add market_cap column to universe table if it doesn't exist."""
    bind = op.get_bind()
    insp = sa.inspect(bind)
    try:
        cols = {c["name"] for c in insp.get_columns("universe")}
    except Exception:
        cols = set()
    if "market_cap" not in cols:
        try:
            with op.batch_alter_table("universe") as batch:
                batch.add_column(sa.Column("market_cap", sa.Float(), nullable=True))
        except Exception:
            pass


def downgrade() -> None:
    """No-op: market_cap exists in the base schema; do not drop."""
    pass
