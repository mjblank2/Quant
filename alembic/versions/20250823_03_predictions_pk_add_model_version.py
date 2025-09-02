"""Change predictions PK to (symbol, ts, model_version)

Revision ID: 20250823_03
Revises: 20250822_02
Create Date: 2025-08-23
"""
from alembic import op
import sqlalchemy as sa

revision = '20250823_03'
down_revision = '20250822_02'
branch_labels = None
depends_on = None

def upgrade():
    # The model_version column already exists in the initial schema,
    # so we just need to update the primary key to include it
    
    # Check if the primary key already includes model_version to avoid duplicate work
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    pk_constraint = inspector.get_pk_constraint("predictions")
    current_pk_columns = pk_constraint.get("constrained_columns", [])
    
    # Only proceed if model_version is not already in the primary key
    if "model_version" not in current_pk_columns:
        # For SQLite, recreate the table with the correct primary key
        op.execute("""
            CREATE TABLE predictions_new (
                symbol VARCHAR(20) NOT NULL,
                ts DATE NOT NULL, 
                y_pred FLOAT NOT NULL,
                model_version VARCHAR(32) NOT NULL DEFAULT 'xgb_v1',
                horizon INTEGER NOT NULL DEFAULT 5,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, ts, model_version)
            )
        """)
        
        # Copy data from old table to new table
        op.execute("""
            INSERT INTO predictions_new (symbol, ts, y_pred, model_version, horizon, created_at)
            SELECT symbol, ts, y_pred, model_version, horizon, created_at FROM predictions
        """)
        
        # Drop old table and rename new table
        op.drop_table("predictions")
        op.execute("ALTER TABLE predictions_new RENAME TO predictions")
        
        # Recreate indexes
        op.create_index("ix_predictions_ts", "predictions", ["ts"])
        op.create_index("ix_predictions_symbol_ts", "predictions", ["symbol", "ts"])
    
    # Create index for model version queries if it doesn't exist
    try:
        op.create_index("ix_predictions_ts_model", "predictions", ["ts", "model_version"], unique=False)
    except Exception:
        pass

def downgrade():
    # Revert to original primary key structure (symbol, ts)
    try:
        # Drop the index first
        op.drop_index("ix_predictions_ts_model", table_name="predictions")
    except Exception:
        pass
    
    # Create a new table with the original primary key (symbol, ts)
    op.execute("""
        CREATE TABLE predictions_new (
            symbol VARCHAR(20) NOT NULL,
            ts DATE NOT NULL, 
            y_pred FLOAT NOT NULL,
            model_version VARCHAR(32) NOT NULL DEFAULT 'xgb_v1',
            horizon INTEGER NOT NULL DEFAULT 5,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, ts)
        )
    """)
    
    # Copy data from current table to new table
    op.execute("""
        INSERT INTO predictions_new (symbol, ts, y_pred, model_version, horizon, created_at)
        SELECT symbol, ts, y_pred, model_version, horizon, created_at FROM predictions
    """)
    
    # Drop old table and rename new table
    op.drop_table("predictions")
    op.execute("ALTER TABLE predictions_new RENAME TO predictions")
    
    # Recreate the original indexes
    op.create_index("ix_predictions_ts", "predictions", ["ts"])
    op.create_index("ix_predictions_symbol_ts", "predictions", ["symbol", "ts"])
