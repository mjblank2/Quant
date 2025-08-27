"""Add bi-temporal support and TimescaleDB optimization

Revision ID: 20250827_10_bitemporal_timescale
Revises: 20250824_000828_add_available_at_to_fundamentals
Create Date: 2025-01-27 16:40:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers
revision = '20250827_10_bitemporal_timescale'
down_revision = '20250824_000828_add_available_at_to_fundamentals'
branch_labels = None
depends_on = None

def upgrade():
    # Add knowledge_date columns for bi-temporal support where missing
    conn = op.get_bind()
    
    # Check if knowledge_date already exists in shares_outstanding
    inspector = sa.inspect(conn)
    shares_cols = [col['name'] for col in inspector.get_columns('shares_outstanding')]
    
    if 'knowledge_date' not in shares_cols:
        op.add_column('shares_outstanding', 
                     sa.Column('knowledge_date', sa.Date(), nullable=True))
        
        # Initialize knowledge_date as as_of + 1 day (default reporting lag)
        conn.execute(text("""
        UPDATE shares_outstanding 
        SET knowledge_date = as_of + INTERVAL '1 day' 
        WHERE knowledge_date IS NULL
        """))
    
    # Add knowledge_date to fundamentals if not exists
    fund_cols = [col['name'] for col in inspector.get_columns('fundamentals')]
    
    if 'knowledge_date' not in fund_cols:
        op.add_column('fundamentals', 
                     sa.Column('knowledge_date', sa.Date(), nullable=True))
        
        # Initialize knowledge_date based on available_at or as_of + reporting lag
        conn.execute(text("""
        UPDATE fundamentals 
        SET knowledge_date = COALESCE(available_at, as_of + INTERVAL '1 day')
        WHERE knowledge_date IS NULL
        """))
    
    # Add data validation metadata table
    op.create_table('data_validation_log',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('run_timestamp', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('validation_type', sa.String(64), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),  # PASSED, FAILED, WARNING
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('metrics', sa.JSON(), nullable=True),
        sa.Column('affected_symbols', sa.ARRAY(sa.String(20)), nullable=True),
    )
    
    op.create_index('ix_validation_log_timestamp', 'data_validation_log', ['run_timestamp'])
    op.create_index('ix_validation_log_type_status', 'data_validation_log', ['validation_type', 'status'])
    
    # Add data lineage tracking table
    op.create_table('data_lineage',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('table_name', sa.String(64), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=True),
        sa.Column('data_date', sa.Date(), nullable=False),
        sa.Column('ingestion_timestamp', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('source', sa.String(32), nullable=False),
        sa.Column('source_timestamp', sa.DateTime(), nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('lineage_metadata', sa.JSON(), nullable=True),
    )
    
    op.create_index('ix_lineage_table_symbol_date', 'data_lineage', ['table_name', 'symbol', 'data_date'])
    op.create_index('ix_lineage_ingestion_time', 'data_lineage', ['ingestion_timestamp'])
    
    # Optimize indexes for better time-series performance
    # These indexes are compatible with both regular PostgreSQL and TimescaleDB
    
    # Enhanced index for daily_bars with covering columns
    try:
        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_daily_bars_symbol_ts_covering 
        ON daily_bars (symbol, ts DESC) 
        INCLUDE (close, volume, adj_close, vwap)
        """))
    except Exception:
        # Fallback for PostgreSQL versions that don't support INCLUDE
        pass
    
    # Index for recent data queries (common in real-time systems)
    conn.execute(text("""
    CREATE INDEX IF NOT EXISTS ix_daily_bars_recent_volume
    ON daily_bars (ts DESC, volume DESC)
    WHERE ts >= CURRENT_DATE - INTERVAL '90 days'
    """))
    
    # Optimize bi-temporal queries
    conn.execute(text("""
    CREATE INDEX IF NOT EXISTS ix_shares_outstanding_bitemporal
    ON shares_outstanding (symbol, as_of DESC, knowledge_date DESC)
    """))
    
    conn.execute(text("""
    CREATE INDEX IF NOT EXISTS ix_fundamentals_bitemporal  
    ON fundamentals (symbol, as_of DESC, knowledge_date DESC)
    """))

def downgrade():
    # Remove new tables
    op.drop_table('data_lineage')
    op.drop_table('data_validation_log')
    
    # Remove knowledge_date columns
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    shares_cols = [col['name'] for col in inspector.get_columns('shares_outstanding')]
    if 'knowledge_date' in shares_cols:
        op.drop_column('shares_outstanding', 'knowledge_date')
    
    fund_cols = [col['name'] for col in inspector.get_columns('fundamentals')]
    if 'knowledge_date' in fund_cols:
        op.drop_column('fundamentals', 'knowledge_date')
    
    # Remove indexes
    op.drop_index('ix_daily_bars_symbol_ts_covering', 'daily_bars', if_exists=True)
    op.drop_index('ix_daily_bars_recent_volume', 'daily_bars', if_exists=True) 
    op.drop_index('ix_shares_outstanding_bitemporal', 'shares_outstanding', if_exists=True)
    op.drop_index('ix_fundamentals_bitemporal', 'fundamentals', if_exists=True)