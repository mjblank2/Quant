"""Create tax_lots table and minor indexes; safe inspector usage.

Revision ID: 20250825_06
Revises: 20250824_05
Create Date: 2025-08-25
"""
from alembic import op
import sqlalchemy as sa

revision = '20250825_06'
down_revision = '20250824_05'
branch_labels = None
depends_on = None

def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    try:
        tables = set(insp.get_table_names())
    except Exception:
        tables = set()

    if 'tax_lots' not in tables:
        op.create_table('tax_lots',
            sa.Column('symbol', sa.String(length=20), nullable=False),
            sa.Column('lot_id', sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column('open_date', sa.Date(), nullable=False),
            sa.Column('shares', sa.Integer(), nullable=False),
            sa.Column('cost_basis', sa.Float(), nullable=False)
        )
        try:
            op.create_index('ix_taxlots_symbol', 'tax_lots', ['symbol'], unique=False)
        except Exception:
            pass

def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)
    try:
        idxs = {i.get('name') for i in insp.get_indexes('tax_lots')}
    except Exception:
        idxs = set()
    if 'ix_taxlots_symbol' in idxs:
        try:
            op.drop_index('ix_taxlots_symbol', table_name='tax_lots')
        except Exception:
            pass
    try:
        op.drop_table('tax_lots')
    except Exception:
        pass
