#!/usr/bin/env python3
"""
Test script to verify the fixed adj_close warning behavior and memory management.
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

def capture_logs(level=logging.WARNING):
    """Helper to capture log output"""
    log_capture_string = StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(level)
    
    # Get the db logger
    logger = logging.getLogger('db')
    logger.setLevel(level)
    logger.addHandler(ch)
    
    return log_capture_string, ch, logger

def test_adj_close_specific_rate_limiting():
    """Test that adj_close warnings are rate-limited more aggressively."""
    print("Testing adj_close specific rate limiting (5 minute intervals)...")
    
    # Create table without adj_close
    with db.engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS daily_bars (
                symbol VARCHAR(20) NOT NULL,
                ts DATE NOT NULL,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume BIGINT,
                vwap FLOAT,
                trade_count INTEGER,
                PRIMARY KEY (symbol, ts)
            )
        """))
        conn.commit()
    
    # Set up log capture for DEBUG level to catch adj_close specific messages
    log_capture, handler, logger = capture_logs(logging.DEBUG)
    
    # Clear cache
    db.clear_table_columns_cache()
    
    print("  Rapid insertions with adj_close column (should be rate limited)...")
    
    # Rapid insertions - should only get one debug message
    for i in range(10):
        df = pd.DataFrame({
            'symbol': [f'TEST{i}'],
            'ts': [date.today() + timedelta(days=i)],
            'open': [100.0 + i],
            'high': [105.0 + i], 
            'low': [95.0 + i],
            'close': [102.0 + i],
            'adj_close': [102.0 + i],  # This column doesn't exist
            'volume': [1000 + i],
            'vwap': [101.0 + i],
            'trade_count': [10 + i]
        })
        
        db.upsert_dataframe(df, db.DailyBar, ['symbol', 'ts'])
        time.sleep(0.1)  # Brief pause
    
    # Clean up
    logger.removeHandler(handler)
    
    # Check logs
    logs = log_capture.getvalue()
    debug_count = logs.count("Dropping adj_close column")
    warning_count = logs.count("Dropping columns not present in daily_bars")
    
    print(f"  Debug messages about adj_close: {debug_count}")
    print(f"  Warning messages: {warning_count}")
    
    # Should only get one debug message due to rate limiting
    success = debug_count <= 1 and warning_count == 0
    if success:
        print("✓ adj_close specific rate limiting working correctly")
    else:
        print("❌ adj_close rate limiting not working properly")
        print(f"  Log content: {logs}")
    
    return success

def test_cache_size_management():
    """Test that the warning cache doesn't grow without bounds."""
    print("\nTesting warning cache size management...")
    
    # Clear cache to start fresh
    db.clear_table_columns_cache()
    
    initial_cache_size = len(db._column_warning_cache)
    print(f"  Initial cache size: {initial_cache_size}")
    
    # Create many different warning scenarios to fill up the cache
    # We'll create warnings for different combinations of missing columns
    for i in range(150):  # More than MAX_CACHE_SIZE (100)
        # Create unique table names and column combinations
        table_name = f"test_table_{i}"
        missing_cols = {f"col_{i % 10}", f"extra_col_{i % 5}"}
        
        # Simulate the warning check (without actually logging)
        db._should_log_column_warning(table_name, missing_cols)
    
    final_cache_size = len(db._column_warning_cache)
    print(f"  Final cache size: {final_cache_size}")
    
    # Cache should be limited to around MAX_CACHE_SIZE (100)
    success = final_cache_size <= 110  # Allow some margin
    if success:
        print(f"✓ Cache size properly managed (kept under control)")
    else:
        print(f"❌ Cache size not managed properly (too large: {final_cache_size})")
    
    return success

def test_mixed_column_warnings():
    """Test that non-adj_close warnings still work normally."""
    print("\nTesting mixed column warning behavior...")
    
    # Create table missing multiple columns
    with db.engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS daily_bars"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS daily_bars (
                symbol VARCHAR(20) NOT NULL,
                ts DATE NOT NULL,
                open FLOAT,
                close FLOAT,
                PRIMARY KEY (symbol, ts)
            )
        """))
        conn.commit()
    
    # Set up log capture
    log_capture, handler, logger = capture_logs(logging.WARNING)
    
    # Clear cache
    db.clear_table_columns_cache()
    
    print("  Testing DataFrame with multiple missing columns...")
    
    df = pd.DataFrame({
        'symbol': ['MIXED1'],
        'ts': [date.today()],
        'open': [100.0],
        'high': [105.0],  # Missing in table
        'low': [95.0],    # Missing in table
        'close': [102.0],
        'adj_close': [102.0],  # Missing in table
        'volume': [1000],      # Missing in table
    })
    
    db.upsert_dataframe(df, db.DailyBar, ['symbol', 'ts'])
    
    # Clean up
    logger.removeHandler(handler)
    
    # Check logs - should get WARNING for multiple columns
    logs = log_capture.getvalue()
    warning_count = logs.count("Dropping columns not present in daily_bars")
    
    print(f"  Warning messages: {warning_count}")
    print(f"  Log content: {logs}")
    
    # Should get a warning since it's not just adj_close
    success = warning_count == 1
    if success:
        print("✓ Mixed column warnings working correctly")
    else:
        print("❌ Mixed column warnings not working properly")
    
    return success

def test_production_simulation():
    """Simulate the production scenario with heavy load."""
    print("\nTesting production-like scenario with heavy load...")
    
    # Create proper schema
    with db.engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS daily_bars"))
        conn.commit()
    
    db.Base.metadata.create_all(db.engine)
    
    # Set up log capture
    log_capture, handler, logger = capture_logs(logging.WARNING)
    
    # Clear cache
    db.clear_table_columns_cache()
    
    print("  Simulating heavy data ingestion load...")
    
    # Simulate heavy load - should work without warnings
    for batch in range(5):  # 5 batches
        batch_data = []
        for i in range(20):  # 20 records per batch
            df = pd.DataFrame({
                'symbol': [f'PROD{batch}_{i}'],
                'ts': [date.today() + timedelta(days=batch*20 + i)],
                'open': [100.0 + i],
                'high': [105.0 + i], 
                'low': [95.0 + i],
                'close': [102.0 + i],
                'adj_close': [102.0 + i],
                'volume': [1000 + i],
                'vwap': [101.0 + i],
                'trade_count': [10 + i]
            })
            
            db.upsert_dataframe(df, db.DailyBar, ['symbol', 'ts'])
        
        print(f"    Batch {batch + 1} completed")
    
    # Clean up
    logger.removeHandler(handler)
    
    # Check for any warnings
    logs = log_capture.getvalue()
    warning_count = logs.count("Dropping columns")
    
    print(f"  Total warnings during heavy load: {warning_count}")
    
    # Should have no warnings in production scenario
    success = warning_count == 0
    if success:
        print("✓ Production scenario working without warnings")
    else:
        print("❌ Production scenario still generating warnings")
        print(f"  Log content: {logs}")
    
    return success

if __name__ == "__main__":
    try:
        print("=" * 70)
        print("TESTING FIXED ADJ_CLOSE WARNING BEHAVIOR AND MEMORY MANAGEMENT")
        print("=" * 70)
        
        test1 = test_adj_close_specific_rate_limiting()
        test2 = test_cache_size_management()
        test3 = test_mixed_column_warnings()
        test4 = test_production_simulation()
        
        print("\n" + "=" * 70)
        if all([test1, test2, test3, test4]):
            print("🎉 ALL TESTS PASSED: adj_close warning spam and memory issues fixed!")
        else:
            print("❌ SOME TESTS FAILED: Issues still need to be addressed")
            
    finally:
        # Cleanup
        if os.path.exists(test_db):
            os.unlink(test_db)