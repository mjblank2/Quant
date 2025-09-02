#!/usr/bin/env python3
"""
Debug script to examine the column cache inspection mechanism for adj_close issue.
"""

import os
import tempfile
from datetime import date, timedelta
import pandas as pd
from sqlalchemy import text
import logging

# Set up test database
test_db = tempfile.mktemp(suffix='.db')
os.environ['DATABASE_URL'] = f'sqlite:///{test_db}'

import db

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
db_logger = logging.getLogger('db')
db_logger.setLevel(logging.DEBUG)

def debug_column_inspection():
    """Debug the column inspection mechanism."""
    print("=" * 60)
    print("DEBUGGING COLUMN INSPECTION MECHANISM")
    print("=" * 60)
    
    # Create database with proper schema including adj_close
    db.Base.metadata.create_all(db.engine)
    
    # Clear cache to start fresh
    db.clear_table_columns_cache()
    
    print("\n1. Initial cache state:")
    print(f"   Cache contents: {db._table_columns_cache}")
    
    # Test column inspection directly
    print("\n2. Testing direct column inspection:")
    with db.engine.connect() as conn:
        columns = db._get_table_columns(conn, db.DailyBar)
        print(f"   Detected columns: {columns}")
        print(f"   adj_close in columns: {'adj_close' in (columns or set())}")
    
    print(f"\n3. Cache after inspection:")
    print(f"   Cache contents: {db._table_columns_cache}")
    
    # Test with a DataFrame containing adj_close
    print("\n4. Testing upsert with adj_close DataFrame:")
    df = pd.DataFrame({
        'symbol': ['DEBUG1'],
        'ts': [date.today()],
        'open': [100.0],
        'high': [105.0], 
        'low': [95.0],
        'close': [102.0],
        'adj_close': [102.0],
        'volume': [1000],
        'vwap': [101.0],
        'trade_count': [10]
    })
    
    print(f"   DataFrame columns: {set(df.columns)}")
    
    # Call upsert - this should work without warnings if cache is correct
    db.upsert_dataframe(df, db.DailyBar, ['symbol', 'ts'])
    
    print(f"\n5. Cache after upsert:")
    print(f"   Cache contents: {db._table_columns_cache}")
    
    # Verify data was inserted
    print("\n6. Verifying data insertion:")
    with db.engine.connect() as conn:
        result = conn.execute(text('SELECT symbol, adj_close FROM daily_bars'))
        rows = result.fetchall()
        print(f"   Inserted rows: {rows}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    try:
        debug_column_inspection()
    finally:
        # Cleanup
        if os.path.exists(test_db):
            os.unlink(test_db)