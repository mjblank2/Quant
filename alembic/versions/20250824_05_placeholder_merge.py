"""Merge heads to create missing parent 20250824_05 (no-op)."""
from alembic import op
import sqlalchemy as sa

revision = '20250824_05'
down_revision = ('20250823_06', '20250824_000828_1_coid', '20250822_03')
branch_labels = None
depends_on = None

def upgrade():
    pass

def downgrade():
    pass
