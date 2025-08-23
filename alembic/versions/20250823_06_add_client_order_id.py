"""add client_order_id to trades

Revision ID: 20250823_06
Revises: 20250823_05
Create Date: 2025-08-23
"""
from alembic import op
import sqlalchemy as sa

revision = '20250823_06'
down_revision = '20250823_05'
branch_labels = None
depends_on = None

def upgrade():
    try:
        with op.batch_alter_table('trades') as batch_op:
            batch_op.add_column(sa.Column('client_order_id', sa.String(length=64), nullable=True))
    except Exception:
        pass

def downgrade():
    try:
        with op.batch_alter_table('trades') as batch_op:
            batch_op.drop_column('client_order_id')
    except Exception:
        pass
