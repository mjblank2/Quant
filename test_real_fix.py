#!/usr/bin/env python3
"""
Test that the fix works correctly with real database.
"""

import pandas as pd
import logging
from datetime import date
from db import DailyBar, upsert_dataframe, _get_table_columns, clear_table_columns_cache, engine

# Set up logging to see all messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def test_real_database_scenario():
    """Test with real database to ensure fix works."""
    
    print("=" * 60)
    print("TESTING REAL DATABASE SCENARIO")
    print("=" * 60)
    
    # Clear cache to start fresh
    clear_table_columns_cache()
    
    # Test data with adj_close
    test_data = pd.DataFrame({
        'symbol': ['REAL_TEST'],
        'ts': [date(2025, 9, 1)],
        'open': [100.0],
        'high': [105.0],
        'low': [99.0],
        'close': [102.0],
        'adj_close': [101.5],  # This should work fine
        'volume': [1000000],
        'vwap': [101.8],
        'trade_count': [1500]
    })
    
    print("\n1. Testing column detection with real database:")
    with engine.begin() as conn:
        columns = _get_table_columns(conn, DailyBar)
        print(f"   Detected columns: {sorted(list(columns)) if columns else None}")
        print(f"   adj_close found: {'adj_close' in columns if columns else False}")
    
    print("\n2. Testing upsert with real database:")
    print(f"   Data columns: {sorted(list(test_data.columns))}")
    
    # This should work without any warnings
    try:
        upsert_dataframe(test_data, DailyBar, ['symbol', 'ts'])
        print("   ✓ Upsert completed successfully")
        
        # Verify data was inserted
        from sqlalchemy import text
        with engine.begin() as conn:
            result = conn.execute(
                text("SELECT symbol, adj_close FROM daily_bars WHERE symbol = 'REAL_TEST'")
            ).fetchall()
            
            if result:
                print(f"   ✓ Data verified: Symbol={result[0][0]}, adj_close={result[0][1]}")
            else:
                print("   ⚠ No data found (might be expected)")
                
    except Exception as e:
        print(f"   ✗ Upsert failed: {e}")

def test_multiple_inserts():
    """Test multiple inserts to see if warnings appear."""
    
    print("\n" + "=" * 60)
    print("TESTING MULTIPLE INSERTS (like production scenario)")
    print("=" * 60)
    
    # Simulate multiple rapid inserts like in production
    for i in range(5):
        test_data = pd.DataFrame({
            'symbol': [f'MULTI_{i}'],
            'ts': [date(2025, 9, 1)],
            'open': [100.0 + i],
            'high': [105.0 + i],
            'low': [99.0 + i],
            'close': [102.0 + i],
            'adj_close': [101.5 + i],
            'volume': [1000000],
            'vwap': [101.8 + i],
            'trade_count': [1500]
        })
        
        print(f"\n{i+1}. Insert #{i+1}:")
        try:
            upsert_dataframe(test_data, DailyBar, ['symbol', 'ts'])
            print("   ✓ Completed")
        except Exception as e:
            print(f"   ✗ Failed: {e}")

if __name__ == "__main__":
    test_real_database_scenario()
    test_multiple_inserts()