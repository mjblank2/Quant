
"""Change predictions PK to (symbol, ts, model_version)

Revision ID: 20250822_03
Revises: 20250822_02
Create Date: 2025-08-22
"""
from alembic import op
import sqlalchemy as sa

revision = '20250822_03'
down_revision = '20250822_02'
branch_labels = None
depends_on = None

def upgrade():
    # Add column if missing
    with op.batch_alter_table('predictions') as batch_op:
        batch_op.add_column(sa.Column('model_version', sa.String(length=32), server_default='xgb_v1'))

    # Drop old PK and create new one including model_version
    op.drop_constraint('predictions_pkey', 'predictions', type_='primary')
    op.create_primary_key('predictions_pkey', 'predictions', ['symbol', 'ts', 'model_version'])

    # Optional: ensure helpful index exists
    op.create_index('ix_predictions_ts_model', 'predictions', ['ts', 'model_version'], unique=False)

def downgrade():
    # Revert to old PK (symbol, ts)
    op.drop_constraint('predictions_pkey', 'predictions', type_='primary')
    op.create_primary_key('predictions_pkey', 'predictions', ['symbol', 'ts'])

    # Drop added index
    op.drop_index('ix_predictions_ts_model', table_name='predictions')

    # Drop column
    with op.batch_alter_table('predictions') as batch_op:
        batch_op.drop_column('model_version')
