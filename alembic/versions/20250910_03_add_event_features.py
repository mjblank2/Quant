"""Add advanced ML feature columns."""

from __future__ import annotations
from alembic import op
import sqlalchemy as sa

revision = "20250910_03"
down_revision = "20250904_02"
branch_labels = None
depends_on = None

def upgrade() -> None:
    for col in [
        sa.Column('reversal_5d_z', sa.Float(), nullable=True),
        sa.Column('ivol_63', sa.Float(), nullable=True),
        sa.Column('beta_63', sa.Float(), nullable=True),
        sa.Column('overnight_gap', sa.Float(), nullable=True),
        sa.Column('illiq_21', sa.Float(), nullable=True),
        sa.Column('fwd_ret', sa.Float(), nullable=True),
        sa.Column('fwd_ret_resid', sa.Float(), nullable=True),
        sa.Column('pead_event', sa.Float(), nullable=True),
        sa.Column('pead_surprise_eps', sa.Float(), nullable=True),
        sa.Column('pead_surprise_rev', sa.Float(), nullable=True),
        sa.Column('russell_inout', sa.Float(), nullable=True),
    ]:
        op.add_column('features', col)


def downgrade() -> None:
    for name in [
        'russell_inout','pead_surprise_rev','pead_surprise_eps','pead_event',
        'fwd_ret_resid','fwd_ret','illiq_21','overnight_gap','beta_63',
        'ivol_63','reversal_5d_z'
    ]:
        op.drop_column('features', name)
