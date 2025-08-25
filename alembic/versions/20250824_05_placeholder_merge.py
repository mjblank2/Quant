"""Merge heads to create missing parent 20250824_05

This file is a no-op that exists solely to unify multiple parallel heads and
satisfy downstream migrations that reference '20250824_05' as their parent.

Heads merged:
- 20250823_06
- 20250824_000828_1_coid
- 20250822_03
"""

from alembic import op
import sqlalchemy as sa

# Alembic identifiers
revision = '20250824_05'
down_revision = ('20250823_06', '20250824_000828_1_coid', '20250822_03')
branch_labels = None
depends_on = None


def upgrade():
    # No schema change; this is a merge-only placeholder
    pass


def downgrade():
    # No schema change; merge-only placeholder
    pass
