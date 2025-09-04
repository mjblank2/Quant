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
    bind = op.get_bind()
    insp = sa.inspect(bind)
    try:
        cols = {c['name'] for c in insp.get_columns('fundamentals')}
    except Exception:
        cols = set()
    if 'available_at' not in cols:
        try:
            with op.batch_alter_table('fundamentals') as batch_op:
                batch_op.add_column(sa.Column('available_at', sa.Date(), nullable=True))
        except Exception:
            pass
    try:
        idxs = {i.get('name') for i in insp.get_indexes('fundamentals')}
    except Exception:
        idxs = set()
    if 'ix_fundamentals_symbol_available_at' not in idxs:
        try:
            op.create_index('ix_fundamentals_symbol_available_at', 'fundamentals', ['symbol','available_at'])
        except Exception:
            pass

def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)
    try:
        idxs = {i.get('name') for i in insp.get_indexes('fundamentals')}
    except Exception:
        idxs = set()
    if 'ix_fundamentals_symbol_available_at' in idxs:
        try:
            op.drop_index('ix_fundamentals_symbol_available_at', table_name='fundamentals')
        except Exception:
            pass
    try:
        cols = {c['name'] for c in insp.get_columns('fundamentals')}
    except Exception:
        cols = set()
    if 'available_at' in cols:
        try:
            with op.batch_alter_table('fundamentals') as batch_op:
                batch_op.drop_column('available_at')
        except Exception:
            pass
