"""Add advanced ML feature columns (idempotent)."""
from __future__ import annotations
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20250910_03"
down_revision = "20250904_02"
branch_labels = None
depends_on = None

COLUMNS_TO_ADD = {
    'reversal_5d_z': sa.Float(),
    'ivol_63': sa.Float(),
    'beta_63': sa.Float(),
    'overnight_gap': sa.Float(),
    'illiq_21': sa.Float(),
    'fwd_ret': sa.Float(),
    'fwd_ret_resid': sa.Float(),
    'pead_event': sa.Float(),
    'pead_surprise_eps': sa.Float(),
    'pead_surprise_rev': sa.Float(),
    'russell_inout': sa.Float(),
}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing = {c['name'] for c in inspector.get_columns('features')}
    for name, coltype in COLUMNS_TO_ADD.items():
        if name not in existing:
            op.add_column('features', sa.Column(name, coltype, nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing = {c['name'] for c in inspector.get_columns('features')}
    for name in reversed(list(COLUMNS_TO_ADD.keys())):
        if name in existing:
            op.drop_column('features', name)
