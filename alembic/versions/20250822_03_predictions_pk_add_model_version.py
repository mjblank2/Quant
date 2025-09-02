"""
Alter predictions primary key to include model version.

Revision ID: 20250822_03
Revises: 20250822_02
Create Date: 2025-08-22
"""

from alembic import op
import sqlalchemy as sa

revision = "20250822_03"
down_revision = "20250822_02"
branch_labels = None
depends_on = None

def upgrade() -> None:
    """Ensure ``model_version`` column exists and update PK."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_cols = [c["name"] for c in inspector.get_columns("predictions")]

    with op.batch_alter_table("predictions") as batch_op:
        if "model_version" not in existing_cols:
            batch_op.add_column(
                sa.Column(
                    "model_version",
                    sa.String(length=32),
                    server_default="xgb_v1",
                )
            )
    try:
        op.drop_constraint("predictions_pkey", "predictions", type_="primary")
    except Exception:
        pass
    op.create_primary_key("predictions_pkey", "predictions", ["id", "model_version"])

def downgrade() -> None:
    """Revert predictions primary key to (symbol, ts)."""
    try:
        op.drop_constraint("predictions_pkey", "predictions", type_="primary")
        op.create_primary_key(
            "predictions_pkey",
            "predictions",
            ["symbol", "ts"],
        )
        op.drop_index("ix_predictions_ts_model", table_name="predictions")
        with op.batch_alter_table("predictions") as batch_op:
            batch_op.drop_column("model_version")
    except Exception:
        pass
