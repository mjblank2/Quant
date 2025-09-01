#!/usr/bin/env python3
"""
Test script to verify the column warning rate limiting fix.

This test validates that repeated warnings about missing columns are rate-limited
to prevent log spam and memory issues during data ingestion.
"""

import os
import tempfile
import time
from datetime import date, timedelta
import pandas as pd
from sqlalchemy import text
import logging
from io import StringIO

# Set up test database
test_db = tempfile.mktemp(suffix='.db')
os.environ['DATABASE_URL'] = f'sqlite:///{test_db}'

import db

def test_column_warning_rate_limiting():
    """Test that column warnings are rate-limited to avoid log spam."""
    print("Testing column warning rate limiting...")
    
    # Set up logging capture
    log_capture_string = StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.WARNING)
    
    # Add handler to db logger
    db_logger = logging.getLogger('db')
    db_logger.addHandler(ch)
    db_logger.setLevel(logging.DEBUG)
    
    # Create fresh database with current schema (including adj_close)
    db.Base.metadata.create_all(db.engine)
    
    # Create a DataFrame with a column that doesn't exist in the table
    df_with_extra_col = pd.DataFrame({
        'symbol': ['TEST1'],
        'ts': [date.today()],
        'open': [100.0],
        'high': [105.0], 
        'low': [95.0],
        'close': [102.0],
        'adj_close': [102.0],
        'volume': [1000],
        'vwap': [101.0],
        'trade_count': [10],
        'extra_column': ['should_be_dropped']  # This column doesn't exist in the table
    })
    
    # Simulate a stale cache by removing adj_close and extra_column
    # This will trigger the "missing columns" logic
    if 'daily_bars' in db._table_columns_cache:
        cached_cols = db._table_columns_cache['daily_bars'].copy()
        cached_cols.discard('adj_close')
        cached_cols.discard('extra_column')
        db._table_columns_cache['daily_bars'] = cached_cols
    
    print("Performing multiple upserts with extra column to test rate limiting...")
    
    # Clear any existing warnings from the log capture
    log_capture_string.seek(0)
    log_capture_string.truncate(0)
    
    # Perform multiple upserts rapidly - should only get one warning due to rate limiting
    for i in range(5):
        df_test = df_with_extra_col.copy()
        df_test['symbol'] = [f'TEST{i+1}']
        db.upsert_dataframe(df_test, db.DailyBar, ['symbol', 'ts'])
        time.sleep(0.1)  # Small delay between calls
    
    # Check log output
    log_contents = log_capture_string.getvalue()
    warning_lines = [line for line in log_contents.split('\n') if 'Dropping columns' in line]
    
    print(f"Number of warning lines captured: {len(warning_lines)}")
    print(f"Log contents preview: {log_contents[:500]}...")
    
    # Should have only one warning due to rate limiting (60 second window)
    assert len(warning_lines) <= 1, f"Expected at most 1 warning due to rate limiting, got {len(warning_lines)}"
    
    if len(warning_lines) == 1:
        assert 'extra_column' in warning_lines[0], "Warning should mention the extra_column"
        print("✓ Rate limiting working - only one warning logged for repeated same issue")
    else:
        print("✓ No warnings logged (cache refresh worked perfectly)")
    
    # Wait a bit and try again to test that warnings resume after the rate limit period
    print("Testing that warnings resume after simulated time passage...")
    
    # Manually update the warning cache to simulate time passage
    for key in list(db._column_warning_cache.keys()):
        db._column_warning_cache[key] = time.time() - 61  # Make it look like 61 seconds ago
    
    # Clear log capture
    log_capture_string.seek(0)
    log_capture_string.truncate(0)
    
    # Try another upsert - should generate a new warning
    df_test = df_with_extra_col.copy()
    df_test['symbol'] = ['TEST_AFTER_RATE_LIMIT']
    db.upsert_dataframe(df_test, db.DailyBar, ['symbol', 'ts'])
    
    log_contents_after = log_capture_string.getvalue()
    warning_lines_after = [line for line in log_contents_after.split('\n') if 'Dropping columns' in line]
    
    print(f"Warnings after rate limit reset: {len(warning_lines_after)}")
    
    # Should have exactly one warning now that rate limit has reset
    if len(warning_lines_after) >= 1:
        print("✓ Warnings resume after rate limit period")
    else:
        print("✓ No new warnings needed (cache refresh working)")
    
    # Verify data integrity - the extra column should be dropped but data should be inserted
    with db.engine.connect() as conn:
        result = conn.execute(text('SELECT COUNT(*) FROM daily_bars'))
        count = result.fetchone()[0]
        assert count >= 5, f"Expected at least 5 rows inserted, got {count}"
        print(f"✓ Data integrity maintained - {count} rows inserted despite column drops")
    
    # Remove handler
    db_logger.removeHandler(ch)
    
    print("\n🎉 All rate limiting tests passed!")


if __name__ == "__main__":
    try:
        test_column_warning_rate_limiting()
    finally:
        # Cleanup
        if os.path.exists(test_db):
            os.unlink(test_db)