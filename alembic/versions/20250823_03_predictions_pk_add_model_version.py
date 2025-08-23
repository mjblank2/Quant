"""Change predictions PK to (symbol, ts, model_version)

Revision ID: 20250823_03
Revises: 20250822_02
Create Date: 2025-08-23
"""
from alembic import op
import sqlalchemy as sa

revision = '20250823_03'
down_revision = '20250822_02'
branch_labels = None
depends_on = None

def upgrade():
    # Ensure model_version exists
    with op.batch_alter_table('predictions') as batch_op:
        try:
            batch_op.add_column(sa.Column('model_version', sa.String(length=32), server_default='xgb_v1'))
        except Exception:
            pass
    # Drop previous PK and create the new one
    try:
        op.drop_constraint('predictions_pkey', 'predictions', type_='primary')
    except Exception:
        pass
    op.create_primary_key('predictions_pkey', 'predictions', ['symbol','ts','model_version'])
    # Helpful index
    try:
        op.create_index('ix_predictions_ts_model', 'predictions', ['ts','model_version'], unique=False)
    except Exception:
        pass

def downgrade():
    try:
        op.drop_constraint('predictions_pkey', 'predictions', type_='primary')
        op.create_primary_key('predictions_pkey', 'predictions', ['symbol','ts'])
        op.drop_index('ix_predictions_ts_model', table_name='predictions')
        with op.batch_alter_table('predictions') as batch_op:
            batch_op.drop_column('model_version')
    except Exception:
        pass
