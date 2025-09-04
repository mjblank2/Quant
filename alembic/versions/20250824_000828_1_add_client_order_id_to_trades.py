"""add client_order_id to trades

Revision ID: 20250824_000828_1_coid
Revises: 20250824_000828_avail
Create Date: 2025-08-24
"""
from alembic import op
import sqlalchemy as sa

revision = '20250824_000828_1_coid'
down_revision = '20250824_000828_avail'
branch_labels = None
depends_on = None

def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)
    try:
        cols = {c['name'] for c in insp.get_columns('trades')}
    except Exception:
        cols = set()
    if 'client_order_id' not in cols:
        try:
            with op.batch_alter_table('trades') as batch_op:
                batch_op.add_column(sa.Column('client_order_id', sa.String(length=64), nullable=True))
        except Exception:
            pass

def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)
    try:
        cols = {c['name'] for c in insp.get_columns('trades')}
    except Exception:
        cols = set()
    if 'client_order_id' in cols:
        try:
            with op.batch_alter_table('trades') as batch_op:
                batch_op.drop_column('client_order_id')
        except Exception:
            pass
