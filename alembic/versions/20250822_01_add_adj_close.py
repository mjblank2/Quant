"""add adj_close to daily_bars

Revision ID: 20250822_01
Revises: 
Create Date: 2025-08-22
"""
from alembic import op
import sqlalchemy as sa

revision = '20250822_01'
down_revision = '20250820_00'
branch_labels = None
depends_on = None

def upgrade():
    with op.batch_alter_table('daily_bars') as batch_op:
        batch_op.add_column(sa.Column('adj_close', sa.Float(), nullable=True))

def downgrade():
    with op.batch_alter_table('daily_bars') as batch_op:
        batch_op.drop_column('adj_close')
