"""add trades (symbol, trade_date) index

Revision ID: 20250823_05
Revises: 20250823_04
Create Date: 2025-08-23
"""
from alembic import op
import sqlalchemy as sa

revision = '20250823_05'
down_revision = '20250823_04'
branch_labels = None
depends_on = None

def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)
    try:
        idxs = {i.get('name') for i in insp.get_indexes('trades')}
    except Exception:
        idxs = set()
    if 'ix_trades_symbol_date' not in idxs:
        try:
            op.create_index('ix_trades_symbol_date', 'trades', ['symbol','trade_date'])
        except Exception:
            pass

def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)
    try:
        idxs = {i.get('name') for i in insp.get_indexes('trades')}
    except Exception:
        idxs = set()
    if 'ix_trades_symbol_date' in idxs:
        try:
            op.drop_index('ix_trades_symbol_date', table_name='trades')
        except Exception:
            pass
