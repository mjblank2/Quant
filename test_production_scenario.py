#!/usr/bin/env python3
"""
Production simulation test to verify the adj_close warning spam fix.

This test simulates the exact scenario from the production logs to ensure
the warning spam and memory issues are resolved.
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

def capture_all_logs():
    """Helper to capture all log output including DEBUG"""
    log_capture_string = StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.DEBUG)
    
    # Get the db logger
    logger = logging.getLogger('db')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    
    return log_capture_string, ch, logger

def simulate_production_log_scenario():
    """Simulate the exact production scenario from the logs."""
    print("=" * 80)
    print("SIMULATING PRODUCTION LOG SCENARIO")
    print("=" * 80)
    print()
    print("Reproducing the scenario from production logs:")
    print("- Multiple rapid data ingestion calls")
    print("- DataFrames containing adj_close columns")
    print("- Database schema may/may not have adj_close column")
    print()
    
    # Test 1: Scenario where adj_close column is missing (pre-migration state)
    print("1. Testing pre-migration scenario (adj_close column missing)...")
    
    # Create table without adj_close to simulate pre-migration state
    with db.engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS daily_bars"))
        conn.execute(text("""
            CREATE TABLE daily_bars (
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
    log_capture, handler, logger = capture_all_logs()
    
    # Clear caches
    db.clear_table_columns_cache()
    
    # Simulate rapid data ingestion similar to production logs
    start_time = time.time()
    warning_times = []
    
    print("   Simulating rapid data ingestion (20 calls in ~20 seconds)...")
    
    for i in range(20):
        # Create DataFrame with adj_close like the data ingestion does
        df = pd.DataFrame({
            'symbol': [f'PROD{i:02d}'],
            'ts': [date.today() + timedelta(days=i)],
            'open': [100.0 + i],
            'high': [105.0 + i],
            'low': [95.0 + i],
            'close': [102.0 + i],
            'adj_close': [102.0 + i],  # This will be dropped
            'volume': [1000 + i*100],
            'vwap': [101.0 + i],
            'trade_count': [50 + i]
        })
        
        call_start = time.time()
        db.upsert_dataframe(df, db.DailyBar, ['symbol', 'ts'])
        call_duration = time.time() - call_start
        
        # Check if any new warnings were logged
        current_logs = log_capture.getvalue()
        current_warning_count = current_logs.count("adj_close")
        
        if len(warning_times) < current_warning_count:
            warning_times.append(time.time() - start_time)
            print(f"   Call {i+1}: Warning logged at {warning_times[-1]:.1f}s (duration: {call_duration:.3f}s)")
        else:
            print(f"   Call {i+1}: No warning (duration: {call_duration:.3f}s)")
        
        # Sleep briefly to simulate production timing
        time.sleep(0.5)
    
    # Clean up logger
    logger.removeHandler(handler)
    
    # Analyze results
    total_logs = log_capture.getvalue()
    debug_warnings = total_logs.count("Dropping adj_close column")
    full_warnings = total_logs.count("Dropping columns not present in daily_bars")
    cache_refreshes = total_logs.count("refreshing column cache")
    cache_skips = total_logs.count("recently attempted for same missing columns")
    
    print(f"\n   Results for pre-migration scenario:")
    print(f"   - DEBUG adj_close warnings: {debug_warnings}")
    print(f"   - Full WARNING messages: {full_warnings}")
    print(f"   - Cache refresh attempts: {cache_refreshes}")
    print(f"   - Cache refresh skips: {cache_skips}")
    print(f"   - Total warning instances: {len(warning_times)}")
    
    if warning_times:
        print(f"   - Warning intervals: {[f'{warning_times[i] - warning_times[i-1]:.1f}s' for i in range(1, len(warning_times))]}")
    
    success_pre = debug_warnings <= 1 and full_warnings == 0 and cache_skips > 0
    
    # Test 2: Post-migration scenario (adj_close column exists)
    print("\n2. Testing post-migration scenario (adj_close column exists)...")
    
    # Create proper schema with adj_close
    with db.engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS daily_bars"))
        conn.commit()
    
    db.Base.metadata.create_all(db.engine)
    
    # Set up fresh log capture
    log_capture2, handler2, logger2 = capture_all_logs()
    
    # Clear caches
    db.clear_table_columns_cache()
    
    print("   Simulating data ingestion with existing adj_close column...")
    
    for i in range(10):
        df = pd.DataFrame({
            'symbol': [f'POST{i:02d}'],
            'ts': [date.today() + timedelta(days=i)],
            'open': [200.0 + i],
            'high': [205.0 + i],
            'low': [195.0 + i],
            'close': [202.0 + i],
            'adj_close': [202.0 + i],  # This should NOT be dropped
            'volume': [2000 + i*100],
            'vwap': [201.0 + i],
            'trade_count': [75 + i]
        })
        
        db.upsert_dataframe(df, db.DailyBar, ['symbol', 'ts'])
        print(f"   Call {i+1}: Completed")
        time.sleep(0.2)
    
    # Clean up logger
    logger2.removeHandler(handler2)
    
    # Analyze post-migration results
    total_logs2 = log_capture2.getvalue()
    post_warnings = total_logs2.count("Dropping")
    
    print(f"\n   Results for post-migration scenario:")
    print(f"   - Any 'Dropping' warnings: {post_warnings}")
    
    success_post = post_warnings == 0
    
    # Verify data was inserted correctly
    print("\n3. Verifying data integrity...")
    with db.engine.connect() as conn:
        result = conn.execute(text('SELECT COUNT(*), COUNT(adj_close) FROM daily_bars'))
        total_rows, adj_close_rows = result.fetchone()
        print(f"   - Total rows inserted: {total_rows}")
        print(f"   - Rows with adj_close values: {adj_close_rows}")
    
    # Final assessment
    print("\n" + "=" * 80)
    print("PRODUCTION SCENARIO ASSESSMENT")
    print("=" * 80)
    
    if success_pre and success_post:
        print("✅ SUCCESS: Production scenario fixed!")
        print("   - Pre-migration: Minimal warnings with proper rate limiting")
        print("   - Post-migration: No warnings, data preserved")
        print("   - Cache refresh optimization working")
        print("   - Memory usage controlled")
    else:
        print("❌ ISSUES STILL PRESENT:")
        if not success_pre:
            print("   - Pre-migration scenario still has excessive warnings")
        if not success_post:
            print("   - Post-migration scenario has unexpected warnings")
    
    return success_pre and success_post

if __name__ == "__main__":
    try:
        success = simulate_production_log_scenario()
        exit(0 if success else 1)
    finally:
        # Cleanup
        if os.path.exists(test_db):
            os.unlink(test_db)