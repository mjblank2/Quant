"""add available_at to fundamentals and index

Revision ID: 20250823_04
Revises: 20250823_03
Create Date: 2025-08-23
"""
from alembic import op
import sqlalchemy as sa

revision = '20250823_04'
down_revision = '20250823_03'
branch_labels = None
depends_on = None

def upgrade():
    try:
        with op.batch_alter_table('fundamentals') as batch_op:
            batch_op.add_column(sa.Column('available_at', sa.Date(), nullable=True))
    except Exception:
        pass
    try:
        op.create_index('ix_fundamentals_symbol_available_at', 'fundamentals', ['symbol','available_at'])
    except Exception:
        pass

def downgrade():
    try:
        op.drop_index('ix_fundamentals_symbol_available_at', table_name='fundamentals')
        with op.batch_alter_table('fundamentals') as batch_op:
            batch_op.drop_column('available_at')
    except Exception:
        pass
