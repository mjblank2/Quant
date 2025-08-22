"""fundamentals + alt_signals + extend features

Revision ID: 20250822_02
Revises: 20250822_01
Create Date: 2025-08-22
"""
from alembic import op
import sqlalchemy as sa

revision = '20250822_02'
down_revision = '20250822_01'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'fundamentals',
        sa.Column('symbol', sa.String(length=20), primary_key=True),
        sa.Column('as_of', sa.Date(), primary_key=True),
        sa.Column('pe_ttm', sa.Float(), nullable=True),
        sa.Column('pb', sa.Float(), nullable=True),
        sa.Column('ps_ttm', sa.Float(), nullable=True),
        sa.Column('debt_to_equity', sa.Float(), nullable=True),
        sa.Column('return_on_assets', sa.Float(), nullable=True),
        sa.Column('gross_margins', sa.Float(), nullable=True),
        sa.Column('profit_margins', sa.Float(), nullable=True),
        sa.Column('current_ratio', sa.Float(), nullable=True),
    )
    op.create_index('ix_fundamentals_symbol_asof', 'fundamentals', ['symbol','as_of'])

    op.create_table(
        'alt_signals',
        sa.Column('symbol', sa.String(length=20), primary_key=True),
        sa.Column('ts', sa.Date(), primary_key=True),
        sa.Column('name', sa.String(length=64), primary_key=True),
        sa.Column('value', sa.Float(), nullable=True),
    )
    op.create_index('ix_alt_symbol_ts', 'alt_signals', ['symbol','ts'])

    with op.batch_alter_table('features') as batch_op:
        batch_op.add_column(sa.Column('f_pe_ttm', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('f_pb', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('f_ps_ttm', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('f_debt_to_equity', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('f_roa', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('f_gm', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('f_profit_margin', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('f_current_ratio', sa.Float(), nullable=True))

def downgrade():
    with op.batch_alter_table('features') as batch_op:
        batch_op.drop_column('f_current_ratio')
        batch_op.drop_column('f_profit_margin')
        batch_op.drop_column('f_gm')
        batch_op.drop_column('f_roa')
        batch_op.drop_column('f_debt_to_equity')
        batch_op.drop_column('f_ps_ttm')
        batch_op.drop_column('f_pb')
        batch_op.drop_column('f_pe_ttm')
    op.drop_index('ix_alt_symbol_ts', table_name='alt_signals')
    op.drop_table('alt_signals')
    op.drop_index('ix_fundamentals_symbol_asof', table_name='fundamentals')
    op.drop_table('fundamentals')

