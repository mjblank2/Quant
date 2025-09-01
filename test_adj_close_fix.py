#!/usr/bin/env python3
"""
Test script to verify the adj_close column cache refresh fix.

This test validates that the upsert_dataframe function correctly handles
the case where the database schema has been updated (e.g., via migration)
but the column cache is stale.
"""

import os
import tempfile
from datetime import date, timedelta
import pandas as pd
from sqlalchemy import text

# Set up test database
test_db = tempfile.mktemp(suffix='.db')
os.environ['DATABASE_URL'] = f'sqlite:///{test_db}'

import db


def test_adj_close_cache_refresh():
    """Test that cache refreshes when adj_close column is detected."""
    print("Testing adj_close column cache refresh fix...")
    
    # Create fresh database with current schema (including adj_close)
    db.Base.metadata.create_all(db.engine)
    
    # Test 1: Normal insertion should work
    df_normal = pd.DataFrame({
        'symbol': ['TEST1'],
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
    
    db.upsert_dataframe(df_normal, db.DailyBar, ['symbol', 'ts'])
    print("✓ Normal insertion with adj_close worked")
    
    # Test 2: Simulate stale cache scenario
    # Remove adj_close from cache to simulate the production issue
    if 'daily_bars' in db._table_columns_cache:
        cached_cols = db._table_columns_cache['daily_bars'].copy()
        cached_cols.discard('adj_close')
        db._table_columns_cache['daily_bars'] = cached_cols
        print(f"  Simulated stale cache (removed adj_close): {len(cached_cols)} columns")
    
    # This insertion should trigger cache refresh without warning
    df_refresh = pd.DataFrame({
        'symbol': ['TEST2'],
        'ts': [date.today() + timedelta(days=1)],
        'open': [200.0],
        'high': [205.0], 
        'low': [195.0],
        'close': [202.0],
        'adj_close': [202.5],  # Should not be dropped
        'volume': [2000],
        'vwap': [201.0],
        'trade_count': [20]
    })
    
    db.upsert_dataframe(df_refresh, db.DailyBar, ['symbol', 'ts'])
    
    # Verify cache was refreshed
    if 'daily_bars' in db._table_columns_cache:
        refreshed_cols = db._table_columns_cache['daily_bars']
        assert 'adj_close' in refreshed_cols, "Cache should contain adj_close after refresh"
        print(f"✓ Cache refreshed successfully: {len(refreshed_cols)} columns including adj_close")
    
    # Test 3: Verify data integrity
    with db.engine.connect() as conn:
        result = conn.execute(text('SELECT symbol, adj_close FROM daily_bars ORDER BY symbol'))
        rows = result.fetchall()
        
        assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
        assert rows[0][1] == 102.0, f"TEST1 adj_close should be 102.0, got {rows[0][1]}"
        assert rows[1][1] == 202.5, f"TEST2 adj_close should be 202.5, got {rows[1][1]}"
        print("✓ Data integrity verified - adj_close values preserved")
    
    print("\n🎉 All tests passed! The fix correctly handles cache refresh for adj_close column.")


if __name__ == "__main__":
    try:
        test_adj_close_cache_refresh()
    finally:
        # Cleanup
        if os.path.exists(test_db):
            os.unlink(test_db)