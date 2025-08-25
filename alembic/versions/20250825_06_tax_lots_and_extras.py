"""Create tax_lots table and minor indexes; safe inspector usage.

Revision ID: 20250825_06
Revises: 20250824_05
Create Date: 2025-08-25
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = '20250825_06'
down_revision = '20250824_05'
branch_labels = None
depends_on = None

def upgrade():
    bind = op.get_bind()
    insp = inspect(bind)

    if 'tax_lots' not in insp.get_table_names():
        op.create_table('tax_lots',
            sa.Column('symbol', sa.String(length=20), nullable=False),
            sa.Column('lot_id', sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column('open_date', sa.Date(), nullable=False),
            sa.Column('shares', sa.Integer(), nullable=False),
            sa.Column('cost_basis', sa.Float(), nullable=False)
        )
        op.create_index('ix_taxlots_symbol', 'tax_lots', ['symbol'], unique=False)

def downgrade():
    op.drop_index('ix_taxlots_symbol', table_name='tax_lots')
    op.drop_table('tax_lots')
