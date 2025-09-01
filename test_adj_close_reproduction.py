#!/usr/bin/env python3
"""
Test to reproduce the adj_close column dropping issue.
"""

import pandas as pd
import logging
from datetime import date
from db import DailyBar, upsert_dataframe, engine, clear_table_columns_cache

# Set up logging to see the warnings
logging.basicConfig(level=logging.WARNING)

def test_adj_close_upsert():
    """Test that adj_close column is properly handled in upsert_dataframe."""
    
    # Clear any existing cache
    clear_table_columns_cache()
    
    # Create test data with adj_close column
    test_data = pd.DataFrame({
        'symbol': ['TEST'],
        'ts': [date(2025, 9, 1)],
        'open': [100.0],
        'high': [105.0],
        'low': [99.0],
        'close': [102.0],
        'adj_close': [101.5],  # This should NOT be dropped
        'volume': [1000000],
        'vwap': [101.8],
        'trade_count': [1500]
    })
    
    print("Test data columns:", list(test_data.columns))
    print("Data being inserted:")
    print(test_data)
    
    # Try to upsert the data
    try:
        upsert_dataframe(test_data, DailyBar, ['symbol', 'ts'])
        print("✓ Upsert completed successfully")
        
        # Verify the data was inserted correctly
        from sqlalchemy import text
        with engine.begin() as conn:
            result = conn.execute(
                text("SELECT symbol, ts, close, adj_close FROM daily_bars WHERE symbol = 'TEST'")
            ).fetchall()
            
            if result:
                print("✓ Data found in database:")
                for row in result:
                    print(f"  Symbol: {row[0]}, Date: {row[1]}, Close: {row[2]}, Adj Close: {row[3]}")
            else:
                print("✗ No data found in database")
                
    except Exception as e:
        print(f"✗ Error during upsert: {e}")

if __name__ == "__main__":
    print("Testing adj_close column handling...")
    print("=" * 50)
    test_adj_close_upsert()