"""Create universe_history, russell_membership, short_borrow, option_overlays.

Revision ID: 20250825_07
Revises: 20250825_06
Create Date: 2025-08-25
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = '20250825_07'
down_revision = '20250825_06'
branch_labels = None
depends_on = None

def upgrade():
    bind = op.get_bind()
    insp = inspect(bind)

    if 'universe_history' not in insp.get_table_names():
        op.create_table('universe_history',
            sa.Column('as_of', sa.Date(), nullable=False),
            sa.Column('symbol', sa.String(length=20), nullable=False),
            sa.Column('included', sa.Boolean(), nullable=False, server_default=sa.sql.expression.true()),
            sa.PrimaryKeyConstraint('as_of', 'symbol')
        )
        op.create_index('ix_universe_history_asof', 'universe_history', ['as_of'])
        op.create_index('ix_universe_history_symbol', 'universe_history', ['symbol'])

    if 'russell_membership' not in insp.get_table_names():
        op.create_table('russell_membership',
            sa.Column('symbol', sa.String(length=20), nullable=False),
            sa.Column('as_of', sa.Date(), nullable=False),
            sa.Column('index_name', sa.String(length=16), nullable=False),
            sa.Column('member', sa.Boolean(), nullable=False),
            sa.PrimaryKeyConstraint('symbol', 'as_of', 'index_name')
        )
        op.create_index('ix_russell_asof', 'russell_membership', ['as_of'])
        op.create_index('ix_russell_symbol', 'russell_membership', ['symbol'])

    if 'short_borrow' not in insp.get_table_names():
        op.create_table('short_borrow',
            sa.Column('symbol', sa.String(length=20), nullable=False),
            sa.Column('ts', sa.Date(), nullable=False),
            sa.Column('available', sa.Boolean(), nullable=False),
            sa.Column('fee_bps', sa.Float(), nullable=True),
            sa.Column('short_interest', sa.Float(), nullable=True),
            sa.Column('source', sa.String(length=32), nullable=True),
            sa.PrimaryKeyConstraint('symbol', 'ts')
        )
        op.create_index('ix_short_borrow_ts', 'short_borrow', ['ts'])
        op.create_index('ix_short_borrow_symbol', 'short_borrow', ['symbol'])

    if 'option_overlays' not in insp.get_table_names():
        op.create_table('option_overlays',
            sa.Column('ts', sa.Date(), nullable=False),
            sa.Column('underlier', sa.String(length=16), nullable=False),
            sa.Column('strategy', sa.String(length=32), nullable=False),
            sa.Column('expiry', sa.Date(), nullable=False),
            sa.Column('k1', sa.Float(), nullable=False),
            sa.Column('k2', sa.Float(), nullable=True),
            sa.Column('notional_pct', sa.Float(), nullable=False),
            sa.Column('est_cost', sa.Float(), nullable=True),
            sa.PrimaryKeyConstraint('ts', 'underlier', 'strategy', 'expiry', 'k1', sa.text('COALESCE(k2, -1)'))
        )
        op.create_index('ix_option_overlays_ts', 'option_overlays', ['ts'])

def downgrade():
    op.drop_index('ix_option_overlays_ts', table_name='option_overlays')
    op.drop_table('option_overlays')
    op.drop_index('ix_short_borrow_symbol', table_name='short_borrow')
    op.drop_index('ix_short_borrow_ts', table_name='short_borrow')
    op.drop_table('short_borrow')
    op.drop_index('ix_russell_symbol', table_name='russell_membership')
    op.drop_index('ix_russell_asof', table_name='russell_membership')
    op.drop_table('russell_membership')
    op.drop_index('ix_universe_history_symbol', table_name='universe_history')
    op.drop_index('ix_universe_history_asof', table_name='universe_history')
    op.drop_table('universe_history')