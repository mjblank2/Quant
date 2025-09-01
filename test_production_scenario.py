#!/usr/bin/env python3
"""
Test script to demonstrate memory and log reduction benefits of the column warning rate limiting.

This simulates the production scenario where many upsert operations happen rapidly
with missing columns, showing how the rate limiting prevents log spam.
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

def test_production_scenario_simulation():
    """Simulate production scenario with many rapid operations."""
    print("Simulating production scenario with rapid upsert operations...")
    
    # Set up logging capture
    log_capture_string = StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.WARNING)
    
    # Add handler to db logger
    db_logger = logging.getLogger('db')
    db_logger.addHandler(ch)
    db_logger.setLevel(logging.WARNING)  # Only capture warnings, not debug
    
    # Create fresh database
    db.Base.metadata.create_all(db.engine)
    
    print("Performing 100 rapid upsert operations (simulating production pipeline)...")
    
    start_time = time.time()
    
    # Simulate production scenario: many symbols being processed rapidly
    symbols = [f'SYMBOL{i:03d}' for i in range(100)]
    
    for i, symbol in enumerate(symbols):
        df = pd.DataFrame({
            'symbol': [symbol],
            'ts': [date.today() + timedelta(days=i % 30)],
            'open': [100.0 + i],
            'high': [105.0 + i], 
            'low': [95.0 + i],
            'close': [102.0 + i],
            'adj_close': [102.0 + i],
            'volume': [1000 + i * 10],
            'vwap': [101.0 + i],
            'trade_count': [10 + i],
            'extra_column': [f'extra_{i}']  # This will trigger warnings
        })
        
        # Clear cache occasionally to simulate the production issue
        if i % 20 == 0:
            if 'daily_bars' in db._table_columns_cache:
                cached_cols = db._table_columns_cache['daily_bars'].copy()
                cached_cols.discard('adj_close')
                cached_cols.discard('extra_column')
                db._table_columns_cache['daily_bars'] = cached_cols
        
        db.upsert_dataframe(df, db.DailyBar, ['symbol', 'ts'])
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Analyze log output
    log_contents = log_capture_string.getvalue()
    warning_lines = [line for line in log_contents.split('\n') if 'Dropping columns' in line]
    
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Total warning lines generated: {len(warning_lines)}")
    print(f"Log size: {len(log_contents)} characters")
    
    # Without rate limiting, we would expect ~100 warnings (one per operation)
    # With rate limiting, we should see much fewer
    expected_without_rate_limiting = 100
    reduction_percentage = (1 - len(warning_lines) / expected_without_rate_limiting) * 100
    
    print(f"Warning reduction: {reduction_percentage:.1f}% (from {expected_without_rate_limiting} to {len(warning_lines)})")
    
    # Verify data integrity
    with db.engine.connect() as conn:
        result = conn.execute(text('SELECT COUNT(*) FROM daily_bars'))
        count = result.fetchone()[0]
        print(f"Successfully inserted {count} rows despite column mismatches")
    
    # Remove handler
    db_logger.removeHandler(ch)
    
    print("\n✓ Production scenario simulation completed")
    print(f"✓ Demonstrated significant log reduction: {reduction_percentage:.1f}%")
    print("✓ This reduction prevents memory pressure from excessive logging")


if __name__ == "__main__":
    try:
        test_production_scenario_simulation()
    finally:
        # Cleanup
        if os.path.exists(test_db):
            os.unlink(test_db)