"""Ensure predictions table matches metadata composite primary key."""

from alembic import op
import sqlalchemy as sa


revision = "20250930_01"
down_revision = "20250928_widen_universe_name"
branch_labels = None
depends_on = None


def _predictions_columns(inspector):
    try:
        cols = inspector.get_columns("predictions")
    except Exception:
        return {}
    return {col["name"]: col for col in cols}


def _predictions_pk(inspector):
    try:
        pk = inspector.get_pk_constraint("predictions")
    except Exception:
        return set()
    return set(pk.get("constrained_columns", []) or [])


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    columns = _predictions_columns(inspector)
    pk_cols = _predictions_pk(inspector)

    expected_pk = {"symbol", "ts", "model_version", "horizon", "created_at"}

    needs_rebuild = False
    if not columns:
        return

    if "id" in columns:
        needs_rebuild = True

    for col in ("horizon", "created_at"):
        if col not in columns or columns[col].get("nullable", True):
            needs_rebuild = True

    if pk_cols != expected_pk:
        needs_rebuild = True

    if not needs_rebuild:
        # Ensure supporting indexes exist in case the table already matches the new schema.
        try:
            op.create_index("ix_predictions_ts", "predictions", ["ts"])
        except Exception:
            pass
        try:
            op.create_index("ix_predictions_ts_model", "predictions", ["ts", "model_version"], unique=False)
        except Exception:
            pass
        return

    op.execute(
        """
        CREATE TABLE predictions_new (
            symbol VARCHAR(20) NOT NULL,
            ts DATE NOT NULL,
            model_version VARCHAR(32) NOT NULL,
            horizon INTEGER NOT NULL DEFAULT 5,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            y_pred FLOAT NOT NULL,
            PRIMARY KEY (symbol, ts, model_version, horizon, created_at)
        )
        """
    )

    select_exprs = [
        "symbol",
        "ts",
        "COALESCE(model_version, 'xgb_v1')",
        "COALESCE(horizon, 5)",
        "COALESCE(created_at, CURRENT_TIMESTAMP)",
        "y_pred",
    ]

    op.execute(
        "INSERT INTO predictions_new (symbol, ts, model_version, horizon, created_at, y_pred) "
        "SELECT {exprs} FROM predictions".format(exprs=", ".join(select_exprs))
    )

    op.drop_table("predictions")
    op.execute("ALTER TABLE predictions_new RENAME TO predictions")

    try:
        op.create_index("ix_predictions_ts", "predictions", ["ts"])
    except Exception:
        pass
    try:
        op.create_index("ix_predictions_ts_model", "predictions", ["ts", "model_version"], unique=False)
    except Exception:
        pass


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "predictions" not in inspector.get_table_names():
        return

    op.execute(
        """
        CREATE TABLE predictions_prev (
            symbol VARCHAR(20) NOT NULL,
            ts DATE NOT NULL,
            model_version VARCHAR(32) NOT NULL,
            y_pred FLOAT NOT NULL,
            horizon INTEGER NOT NULL DEFAULT 5,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, ts, model_version)
        )
        """
    )

    op.execute(
        """
        INSERT INTO predictions_prev (symbol, ts, model_version, y_pred, horizon, created_at)
        SELECT symbol, ts, model_version, y_pred, horizon, created_at
        FROM (
            SELECT symbol,
                   ts,
                   model_version,
                   horizon,
                   created_at,
                   y_pred,
                   ROW_NUMBER() OVER (
                       PARTITION BY symbol, ts, model_version
                       ORDER BY created_at DESC
                   ) AS rn
            FROM predictions
        ) ranked
        WHERE rn = 1
        """
    )

    op.drop_table("predictions")
    op.execute("ALTER TABLE predictions_prev RENAME TO predictions")

    try:
        op.create_index("ix_predictions_ts", "predictions", ["ts"])
    except Exception:
        pass
    try:
        op.create_index("ix_predictions_ts_model", "predictions", ["ts", "model_version"], unique=False)
    except Exception:
        pass

