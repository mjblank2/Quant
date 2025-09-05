"""Ensure features table has primary key on (symbol, ts).

Revision ID: 20250904_02
Revises: 20250903_01
Create Date: 2025-09-04
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20250904_02"
down_revision = "20250903_01"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'features_pkey'
                  AND conrelid = 'features'::regclass
            ) THEN
                ALTER TABLE features ADD CONSTRAINT features_pkey PRIMARY KEY (symbol, ts);
            END IF;
        END$$;
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE features DROP CONSTRAINT IF EXISTS features_pkey;")
