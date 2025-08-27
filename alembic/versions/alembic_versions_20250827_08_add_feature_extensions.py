"""Add missing feature extension columns used by v16+ feature builder.

Revision ID: 20250827_08
Revises: 20250825_07
Create Date: 2025-08-27
"""
from alembic import op
import sqlalchemy as sa

revision = "20250827_08"
down_revision = "20250825_07"
branch_labels = None
depends_on = None

def upgrade():
    try:
        with op.batch_alter_table("features") as batch_op:
            try:
                batch_op.add_column(sa.Column("adv_usd_21", sa.Float(), nullable=True))
            except Exception:
                pass
            try:
                batch_op.add_column(sa.Column("overnight_gap", sa.Float(), nullable=True))
            except Exception:
                pass
            try:
                batch_op.add_column(sa.Column("illiq_21", sa.Float(), nullable=True))
            except Exception:
                pass
            try:
                batch_op.add_column(sa.Column("beta_63", sa.Float(), nullable=True))
            except Exception:
                pass
    except Exception:
        pass

def downgrade():
    try:
        with op.batch_alter_table("features") as batch_op:
            for col in ["beta_63","illiq_21","overnight_gap","adv_usd_21"]:
                try:
                    batch_op.drop_column(col)
                except Exception:
                    pass
    except Exception:
        pass