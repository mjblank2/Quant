"""Merge multiple heads: v17 fast-follows and task status/market cap branches.

Revision ID: 20250902_merge_heads
Revises: 20250825_07, 20250827_12
Create Date: 2025-09-02

"""
from alembic import op
import sqlalchemy as sa

revision = '20250902_merge_heads'
down_revision = ('20250825_07', '20250827_12')
branch_labels = None
depends_on = None

def upgrade():
    """Merge the two divergent migration branches.
    
    This is a no-op merge that combines:
    - 20250825_07: v17 fast-follows (universe_history, russell_membership, etc.)
    - 20250827_12: market_cap column addition to universe table
    """
    pass

def downgrade():
    """No operations needed for downgrade."""
    pass