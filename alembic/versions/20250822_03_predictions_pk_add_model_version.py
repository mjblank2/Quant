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
    """Ensure model_version is part of the PK in a safe, idempotent way."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    try:
        pk_constraint = inspector.get_pk_constraint("predictions")
        current_pk_columns = pk_constraint.get("constrained_columns", []) or []
    except Exception:
        current_pk_columns = []

    if "model_version" not in current_pk_columns:
        op.execute("""
            CREATE TABLE predictions_new (
                symbol VARCHAR(20) NOT NULL,
                ts DATE NOT NULL, 
                y_pred FLOAT NOT NULL,
                model_version VARCHAR(32) NOT NULL DEFAULT 'xgb_v1',
                horizon INTEGER NOT NULL DEFAULT 5,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, ts, model_version)
            )
        """)
        op.execute("""
            INSERT INTO predictions_new (symbol, ts, y_pred, model_version, horizon, created_at)
            SELECT symbol, ts, y_pred, model_version, horizon, created_at FROM predictions
        """)
        op.drop_table("predictions")
        op.execute("ALTER TABLE predictions_new RENAME TO predictions")

        # Recreate helpful indexes (ignore if already present)
        try:
            op.create_index("ix_predictions_ts", "predictions", ["ts"])
        except Exception:
            pass
        try:
            op.create_index("ix_predictions_symbol_ts", "predictions", ["symbol", "ts"])
        except Exception:
            pass
        try:
            op.create_index("ix_predictions_ts_model", "predictions", ["ts", "model_version"])
        except Exception:
            pass
    else:
        # Ensure ts+model_version index exists
        try:
            op.create_index("ix_predictions_ts_model", "predictions", ["ts", "model_version"])
        except Exception:
            pass

def downgrade() -> None:
    """Revert predictions primary key to (symbol, ts)."""
    try:
        op.drop_index("ix_predictions_ts_model", table_name="predictions")
    except Exception:
        pass

    op.execute("""
        CREATE TABLE predictions_new (
            symbol VARCHAR(20) NOT NULL,
            ts DATE NOT NULL, 
            y_pred FLOAT NOT NULL,
            model_version VARCHAR(32) NOT NULL DEFAULT 'xgb_v1',
            horizon INTEGER NOT NULL DEFAULT 5,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, ts)
        )
    """)
    op.execute("""
        INSERT INTO predictions_new (symbol, ts, y_pred, model_version, horizon, created_at)
        SELECT symbol, ts, y_pred, model_version, horizon, created_at FROM predictions
    """)
    op.drop_table("predictions")
    op.execute("ALTER TABLE predictions_new RENAME TO predictions")

    try:
        op.create_index("ix_predictions_ts", "predictions", ["ts"])
    except Exception:
        pass
    try:
        op.create_index("ix_predictions_symbol_ts", "predictions", ["symbol", "ts"])
    except Exception:
        pass
