#!/usr/bin/env python3
"""
Test script to reproduce the adj_close warning spam issue that causes memory problems.

This test validates that the rate limiting in _should_log_column_warning is working
correctly and prevents excessive logging that could lead to memory issues.
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

def capture_logs():
    """Helper to capture log output"""
    log_capture_string = StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.WARNING)
    
    # Get the db logger
    logger = logging.getLogger('db')
    logger.setLevel(logging.WARNING)
    logger.addHandler(ch)
    
    return log_capture_string, ch, logger

def test_memory_warning_spam():
    """Test that reproduces the adj_close warning spam issue."""
    print("Testing adj_close warning spam that causes memory issues...")
    
    # Create fresh database without adj_close column (simulate pre-migration state)
    # We'll manually create the schema without adj_close to simulate the issue
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
    
    # Set up log capture
    log_capture, handler, logger = capture_logs()
    
    # Clear any existing cache to simulate fresh start
    db.clear_table_columns_cache()
    
    print("  Simulating rapid data ingestion with adj_close columns...")
    
    # Simulate rapid data ingestion that would trigger warnings
    warning_count = 0
    for i in range(20):  # Simulate 20 rapid calls
        df = pd.DataFrame({
            'symbol': [f'TEST{i}'],
            'ts': [date.today() + timedelta(days=i)],
            'open': [100.0 + i],
            'high': [105.0 + i], 
            'low': [95.0 + i],
            'close': [102.0 + i],
            'adj_close': [102.0 + i],  # This column doesn't exist in our test table
            'volume': [1000 + i],
            'vwap': [101.0 + i],
            'trade_count': [10 + i]
        })
        
        db.upsert_dataframe(df, db.DailyBar, ['symbol', 'ts'])
        
        # Check log output after each call
        logs = log_capture.getvalue()
        current_warning_count = logs.count("Dropping columns not present in daily_bars: {'adj_close'}")
        
        if current_warning_count > warning_count:
            warning_count = current_warning_count
            print(f"    Warning #{warning_count} at iteration {i+1}")
        
        # Sleep briefly to simulate time passing (but less than rate limit)
        time.sleep(0.1)
    
    # Clean up
    logger.removeHandler(handler)
    
    # Analyze results
    total_logs = log_capture.getvalue()
    final_warning_count = total_logs.count("Dropping columns not present in daily_bars: {'adj_close'}")
    
    print(f"\n  Total warnings logged: {final_warning_count}")
    print(f"  Total iterations: 20")
    print(f"  Rate limiting effectiveness: {100 - (final_warning_count/20)*100:.1f}%")
    
    # The rate limiting should prevent most warnings after the first one
    if final_warning_count > 3:  # Allow for some margin, but should be much less than 20
        print(f"❌ ISSUE REPRODUCED: Excessive warnings ({final_warning_count}) indicate rate limiting is not working properly")
        return False
    else:
        print(f"✓ Rate limiting working properly: Only {final_warning_count} warnings for 20 operations")
        return True

def test_production_scenario():
    """Test the exact production scenario with adj_close column existing."""
    print("\nTesting production scenario where adj_close column exists...")
    
    # Drop the old table first to ensure clean state
    with db.engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS daily_bars"))
        conn.commit()
    
    # Create database with proper schema including adj_close
    db.Base.metadata.create_all(db.engine)
    
    # Set up log capture
    log_capture, handler, logger = capture_logs()
    
    # Clear cache to simulate fresh start
    db.clear_table_columns_cache()
    
    print("  Simulating data ingestion with existing adj_close column...")
    
    # Simulate the production scenario - data ingestion should work without warnings
    for i in range(10):
        df = pd.DataFrame({
            'symbol': [f'PROD{i}'],
            'ts': [date.today() + timedelta(days=i)],
            'open': [100.0 + i],
            'high': [105.0 + i], 
            'low': [95.0 + i],
            'close': [102.0 + i],
            'adj_close': [102.0 + i],  # This column DOES exist in the schema
            'volume': [1000 + i],
            'vwap': [101.0 + i],
            'trade_count': [10 + i]
        })
        
        db.upsert_dataframe(df, db.DailyBar, ['symbol', 'ts'])
    
    # Clean up
    logger.removeHandler(handler)
    
    # Check for any warnings
    total_logs = log_capture.getvalue()
    warning_count = total_logs.count("Dropping columns not present in daily_bars")
    
    print(f"  Warnings in production scenario: {warning_count}")
    
    if warning_count > 0:
        print(f"❌ ISSUE: Unexpected warnings in production scenario with existing adj_close column")
        print(f"  Log output: {total_logs}")
        return False
    else:
        print("✓ Production scenario working correctly - no warnings")
        return True

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("TESTING ADJ_CLOSE WARNING SPAM AND MEMORY ISSUES")
        print("=" * 60)
        
        test1_result = test_memory_warning_spam()
        test2_result = test_production_scenario()
        
        print("\n" + "=" * 60)
        if test1_result and test2_result:
            print("✅ ALL TESTS PASSED: No memory-causing warning spam detected")
        else:
            print("❌ ISSUES DETECTED: Warning spam may be causing memory problems")
            
    finally:
        # Cleanup
        if os.path.exists(test_db):
            os.unlink(test_db)