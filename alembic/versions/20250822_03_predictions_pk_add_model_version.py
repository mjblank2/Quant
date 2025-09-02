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
    # For SQLite, the safest way to change a primary key is to recreate the table
    # Since batch mode in Alembic handles this automatically, we just need to 
    # define the new structure
    
    with op.batch_alter_table("predictions", recreate="always") as batch_op:
        # The table will be recreated with model_version as part of the primary key
        # We need to ensure the primary key structure is correct
        pass
    
    # After recreating the table, make sure the primary key is correctly set
    # We'll do this by checking the table structure and recreating if needed
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    
    # Check current primary key
    pk_constraint = inspector.get_pk_constraint("predictions")
    current_pk_columns = pk_constraint.get("constrained_columns", [])
    
    # If the primary key doesn't include model_version, we need to fix it
    if "model_version" not in current_pk_columns:
        # Create a new table with the correct primary key
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
        op.create_index("ix_predictions_ts_model", "predictions", ["ts", "model_version"])

def downgrade() -> None:
    """Revert predictions primary key to (symbol, ts)."""
    # For SQLite, recreate the table with the original primary key structure
    # Drop the index first
    try:
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
