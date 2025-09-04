"""v17 fast-follows: universe_history, russell_membership, short_borrow, option_overlays, trades.order_metadata.

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
    try:
        tables = set(insp.get_table_names())
    except Exception:
        tables = set()

    if 'universe_history' not in tables:
        op.create_table('universe_history',
            sa.Column('as_of', sa.Date(), nullable=False),
            sa.Column('symbol', sa.String(length=20), nullable=False),
            sa.PrimaryKeyConstraint('as_of', 'symbol')
        )
        try:
            op.create_index('ix_universe_hist_asof', 'universe_history', ['as_of'], unique=False)
        except Exception:
            pass
        try:
            op.create_index('ix_universe_hist_symbol', 'universe_history', ['symbol'], unique=False)
        except Exception:
            pass

    if 'russell_membership' not in tables:
        op.create_table('russell_membership',
            sa.Column('symbol', sa.String(length=20), nullable=False),
            sa.Column('ts', sa.Date(), nullable=False),
            sa.Column('action', sa.String(length=8), nullable=False),
            sa.PrimaryKeyConstraint('symbol','ts')
        )
        try:
            op.create_index('ix_russell_ts', 'russell_membership', ['ts'], unique=False)
        except Exception:
            pass

    if 'short_borrow' not in tables:
        op.create_table('short_borrow',
            sa.Column('symbol', sa.String(length=20), nullable=False),
            sa.Column('ts', sa.Date(), nullable=False),
            sa.Column('available', sa.Integer(), nullable=True),
            sa.Column('fee_bps', sa.Float(), nullable=True),
            sa.Column('short_interest', sa.Float(), nullable=True),
            sa.Column('source', sa.String(length=32), nullable=True),
            sa.PrimaryKeyConstraint('symbol','ts')
        )
        try:
            op.create_index('ix_borrow_symbol_ts', 'short_borrow', ['symbol','ts'], unique=False)
        except Exception:
            pass

    if 'option_overlays' not in tables:
        op.create_table('option_overlays',
            sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column('as_of', sa.Date(), nullable=False),
            sa.Column('symbol', sa.String(length=20), nullable=False),
            sa.Column('strategy', sa.String(length=16), nullable=False),
            sa.Column('tenor_days', sa.Integer(), nullable=False),
            sa.Column('put_strike', sa.Float(), nullable=True),
            sa.Column('call_strike', sa.Float(), nullable=True),
            sa.Column('est_premium_pct', sa.Float(), nullable=True),
            sa.Column('notes', sa.String(length=256), nullable=True),
        )
        try:
            op.create_index('ix_option_overlays_asof', 'option_overlays', ['as_of'], unique=False)
        except Exception:
            pass

    # Add order_metadata JSON to trades if missing
    if 'trades' in tables:
        try:
            cols = {c['name'] for c in insp.get_columns('trades')}
        except Exception:
            cols = set()
        if 'order_metadata' not in cols:
            try:
                with op.batch_alter_table('trades') as batch_op:
                    batch_op.add_column(sa.Column('order_metadata', sa.JSON(), nullable=True))
            except Exception:
                try:
                    op.add_column('trades', sa.Column('order_metadata', sa.JSON(), nullable=True))
                except Exception:
                    pass

def downgrade():
    bind = op.get_bind()
    insp = inspect(bind)
    try:
        tables = set(insp.get_table_names())
    except Exception:
        tables = set()

    if 'trades' in tables:
        try:
            cols = {c['name'] for c in insp.get_columns('trades')}
        except Exception:
            cols = set()
        if 'order_metadata' in cols:
            try:
                with op.batch_alter_table('trades') as batch_op:
                    batch_op.drop_column('order_metadata')
            except Exception:
                pass

    if 'option_overlays' in tables:
        try:
            idxs = {i.get('name') for i in insp.get_indexes('option_overlays')}
        except Exception:
            idxs = set()
        if 'ix_option_overlays_asof' in idxs:
            try:
                op.drop_index('ix_option_overlays_asof', table_name='option_overlays')
            except Exception:
                pass
        try:
            op.drop_table('option_overlays')
        except Exception:
            pass

    if 'short_borrow' in tables:
        try:
            idxs = {i.get('name') for i in insp.get_indexes('short_borrow')}
        except Exception:
            idxs = set()
        if 'ix_borrow_symbol_ts' in idxs:
            try:
                op.drop_index('ix_borrow_symbol_ts', table_name='short_borrow')
            except Exception:
                pass
        try:
            op.drop_table('short_borrow')
        except Exception:
            pass

    if 'russell_membership' in tables:
        try:
            idxs = {i.get('name') for i in insp.get_indexes('russell_membership')}
        except Exception:
            idxs = set()
        if 'ix_russell_ts' in idxs:
            try:
                op.drop_index('ix_russell_ts', table_name='russell_membership')
            except Exception:
                pass
        try:
            op.drop_table('russell_membership')
        except Exception:
            pass

    if 'universe_history' in tables:
        try:
            idxs = {i.get('name') for i in insp.get_indexes('universe_history')}
        except Exception:
            idxs = set()
        if 'ix_universe_hist_symbol' in idxs:
            try:
                op.drop_index('ix_universe_hist_symbol', table_name='universe_history')
            except Exception:
                pass
        if 'ix_universe_hist_asof' in idxs:
            try:
                op.drop_index('ix_universe_hist_asof', table_name='universe_history')
            except Exception:
                pass
        try:
            op.drop_table('universe_history')
        except Exception:
            pass
