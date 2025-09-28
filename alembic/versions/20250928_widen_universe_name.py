"""Widen universe.name from VARCHAR(128) to VARCHAR(256)

Revision ID: 20250928_widen_universe_name
Revises: 20250827_12
Create Date: 2025-09-28

"""

from alembic import op
import sqlalchemy as sa


revision = "20250928_widen_universe_name"
down_revision = "20250910_04"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Widen universe.name column from VARCHAR(128) to VARCHAR(256)."""
    bind = op.get_bind()
    insp = sa.inspect(bind)
    
    try:
        cols = {c["name"]: c for c in insp.get_columns("universe")}
    except Exception:
        cols = {}
    
    # Check if the universe table and name column exist
    if "name" in cols:
        try:
            # Use batch alter table for SQLite compatibility
            with op.batch_alter_table("universe") as batch:
                # Alter the name column to VARCHAR(256)
                batch.alter_column(
                    "name",
                    existing_type=sa.String(128),
                    type_=sa.String(256),
                    existing_nullable=True
                )
        except Exception as e:
            # Log but don't fail the migration - this is defensive
            print(f"Warning: Could not widen universe.name column: {e}")


def downgrade() -> None:
    """Downgrade by narrowing universe.name back to VARCHAR(128)."""
    bind = op.get_bind()
    insp = sa.inspect(bind)
    
    try:
        cols = {c["name"]: c for c in insp.get_columns("universe")}
    except Exception:
        cols = {}
    
    # Check if the universe table and name column exist
    if "name" in cols:
        try:
            with op.batch_alter_table("universe") as batch:
                batch.alter_column(
                    "name",
                    existing_type=sa.String(256),
                    type_=sa.String(128),
                    existing_nullable=True
                )
        except Exception as e:
            print(f"Warning: Could not narrow universe.name column: {e}")