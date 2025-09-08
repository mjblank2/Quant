"""Add reversal, ivol, and forward return columns to features table.

Revision ID: 20250905_01
Revises: 20250904_02
Create Date: 2025-09-05
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20250905_01"
down_revision = "20250904_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("features") as batch_op:
        batch_op.add_column(sa.Column("reversal_5d_z", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("ivol_63", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("fwd_ret", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("fwd_ret_resid", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("features") as batch_op:
        batch_op.drop_column("fwd_ret_resid")
        batch_op.drop_column("fwd_ret")
        batch_op.drop_column("ivol_63")
        batch_op.drop_column("reversal_5d_z")
