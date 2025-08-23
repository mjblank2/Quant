"""initial schema

Revision ID: 20250820_00
Revises:
Create Date: 2025-08-20
"""
from alembic import op
import sqlalchemy as sa

revision = '20250820_00'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'daily_bars',
        sa.Column('symbol', sa.String(20), primary_key=True),
        sa.Column('ts', sa.Date(), primary_key=True),
        sa.Column('open', sa.Float(), nullable=False),
        sa.Column('high', sa.Float(), nullable=False),
        sa.Column('low', sa.Float(), nullable=False),
        sa.Column('close', sa.Float(), nullable=False),
        sa.Column('volume', sa.BigInteger(), nullable=False),
        sa.Column('vwap', sa.Float(), nullable=True),
        sa.Column('trade_count', sa.Integer(), nullable=True),
    )
    op.create_index('ix_daily_bars_symbol_ts', 'daily_bars', ['symbol','ts'])

    op.create_table(
        'universe',
        sa.Column('symbol', sa.String(20), primary_key=True),
        sa.Column('name', sa.String(128), nullable=True),
        sa.Column('exchange', sa.String(12), nullable=True),
        sa.Column('market_cap', sa.Float(), nullable=True),
        sa.Column('adv_usd_20', sa.Float(), nullable=True),
        sa.Column('included', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('last_updated', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'))
    )

    op.create_table(
        'features',
        sa.Column('symbol', sa.String(20), primary_key=True),
        sa.Column('ts', sa.Date(), primary_key=True),
        sa.Column('ret_1d', sa.Float(), nullable=True),
        sa.Column('ret_5d', sa.Float(), nullable=True),
        sa.Column('ret_21d', sa.Float(), nullable=True),
        sa.Column('mom_21', sa.Float(), nullable=True),
        sa.Column('mom_63', sa.Float(), nullable=True),
        sa.Column('vol_21', sa.Float(), nullable=True),
        sa.Column('rsi_14', sa.Float(), nullable=True),
        sa.Column('turnover_21', sa.Float(), nullable=True),
        sa.Column('size_ln', sa.Float(), nullable=True),
    )
    op.create_index('ix_features_symbol_ts', 'features', ['symbol','ts'])

    op.create_table(
        'predictions',
        sa.Column('symbol', sa.String(20), primary_key=True),
        sa.Column('ts', sa.Date(), primary_key=True),
        sa.Column('y_pred', sa.Float(), nullable=False),
        sa.Column('model_version', sa.String(32), nullable=False, server_default='xgb_v1'),
        sa.Column('horizon', sa.Integer(), nullable=False, server_default='5'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'))
    )
    op.create_index('ix_predictions_ts', 'predictions', ['ts'])
    op.create_index('ix_predictions_symbol_ts', 'predictions', ['symbol','ts'])
    op.create_index('ix_predictions_ts_model', 'predictions', ['ts','model_version'])

    op.create_table(
        'positions',
        sa.Column('ts', sa.Date(), primary_key=True),
        sa.Column('symbol', sa.String(20), primary_key=True),
        sa.Column('weight', sa.Float(), nullable=False),
        sa.Column('price', sa.Float(), nullable=True),
        sa.Column('shares', sa.Integer(), nullable=True),
    )
    op.create_index('ix_positions_ts_symbol', 'positions', ['ts','symbol'])

    op.create_table(
        'trades',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('ts', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('trade_date', sa.Date(), nullable=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('side', sa.String(4), nullable=False),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('price', sa.Float(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='generated'),
        sa.Column('broker_order_id', sa.String(64), nullable=True),
    )
    op.create_index('ix_trades_status_id', 'trades', ['status','id'])
    op.create_index('ix_trades_symbol_date', 'trades', ['symbol','trade_date'])

    op.create_table(
        'backtest_equity',
        sa.Column('ts', sa.Date(), primary_key=True),
        sa.Column('equity', sa.Float(), nullable=False),
        sa.Column('daily_return', sa.Float(), nullable=False),
        sa.Column('drawdown', sa.Float(), nullable=False),
    )

def downgrade():
    for t in ['backtest_equity','trades','positions','predictions','features','universe','daily_bars']:
        try:
            op.drop_table(t)
        except Exception:
            pass
