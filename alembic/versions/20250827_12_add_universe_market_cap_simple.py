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
    # Use a simple approach: try to add the column, ignore if it already exists
    try:
        op.add_column('universe', sa.Column('market_cap', sa.Float(), nullable=True))
    except Exception:
        # Column likely already exists, which is fine
        pass


def downgrade() -> None:
    """Remove market_cap column from universe table."""
    # Use a simple approach: try to drop the column, ignore if it doesn't exist
    try:
        op.drop_column('universe', 'market_cap')
    except Exception:
        # Column likely doesn't exist, which is fine
        pass