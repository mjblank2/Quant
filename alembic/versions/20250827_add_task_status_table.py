"""add_task_status_table

Revision ID: 20250827_add_task_status_table
Revises: 20250827_10
Create Date: 2025-08-27 16:30:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = "20250827_add_task_status_table"
down_revision = "20250827_10"
branch_labels = None
depends_on = None


def upgrade():
    """Create task_status table if it does not already exist."""
    bind = op.get_bind()
    inspector = inspect(bind)

    try:
        has_table = inspector.has_table("task_status")
    except Exception:
        has_table = False

    if not has_table:
        op.create_table(
            "task_status",
            sa.Column("task_id", sa.String(length=128), nullable=False),
            sa.Column("task_name", sa.String(length=128), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.Column("result", sa.JSON(), nullable=True),
            sa.Column("error_message", sa.String(length=1024), nullable=True),
            sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
            sa.PrimaryKeyConstraint("task_id"),
        )

    try:
        inspector = inspect(bind)
        indexes = {idx.get("name") for idx in inspector.get_indexes("task_status")}
    except Exception:
        indexes = set()
    if "ix_task_status_created_at" not in indexes:
        try:
            op.create_index(
                "ix_task_status_created_at",
                "task_status",
                ["created_at"],
                unique=False,
            )
        except Exception:
            pass


def downgrade():
    """Drop task_status table and related index if they exist."""
    bind = op.get_bind()
    inspector = inspect(bind)

    try:
        has_table = inspector.has_table("task_status")
    except Exception:
        has_table = False

    if has_table:
        try:
            indexes = {idx.get("name") for idx in inspector.get_indexes("task_status")}
        except Exception:
            indexes = set()
        if "ix_task_status_created_at" in indexes:
            try:
                op.drop_index("ix_task_status_created_at", table_name="task_status")
            except Exception:
                pass
        try:
            op.drop_table("task_status")
        except Exception:
            pass
