
"""add available_at to fundamentals

Revision ID: 20250824_000828_avail
Revises: 20250823_03
Create Date: 2025-08-24
"""
from alembic import op
import sqlalchemy as sa

revision = '20250824_000828_avail'
down_revision = '20250823_03'
branch_labels = None
depends_on = None

def upgrade():
    try:
        with op.batch_alter_table('fundamentals') as batch_op:
            batch_op.add_column(sa.Column('available_at', sa.Date(), nullable=True))
    except Exception:
        pass

def downgrade():
    try:
        with op.batch_alter_table('fundamentals') as batch_op:
            batch_op.drop_column('available_at')
    except Exception:
        pass
