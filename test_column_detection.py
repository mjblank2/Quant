#!/usr/bin/env python3
"""
Test to diagnose the table column detection issue.
"""

import pandas as pd
import logging
from datetime import date
from db import DailyBar, _get_table_columns, _table_columns_cache, engine, clear_table_columns_cache
from sqlalchemy import text

# Set up logging to see all messages
logging.basicConfig(level=logging.DEBUG)

def test_column_detection():
    """Test the column detection mechanism."""
    
    print("=" * 60)
    print("TESTING COLUMN DETECTION MECHANISM")
    print("=" * 60)
    
    # Clear cache first
    clear_table_columns_cache()
    print(f"Cache cleared. Current cache: {_table_columns_cache}")
    
    with engine.begin() as conn:
        # Test 1: Direct column detection
        print("\n1. Testing direct column detection:")
        columns = _get_table_columns(conn, DailyBar)
        print(f"   Detected columns: {sorted(list(columns)) if columns else None}")
        print(f"   Cache after detection: {_table_columns_cache}")
        
        # Test 2: Check if adj_close is in the detected columns
        print("\n2. Checking for adj_close:")
        if columns:
            has_adj_close = 'adj_close' in columns
            print(f"   adj_close found: {has_adj_close}")
        else:
            print("   Cannot check - no columns detected")
            
        # Test 3: Manual verification with database query
        print("\n3. Manual database verification:")
        try:
            # Check SQLite schema
            schema_result = conn.execute(text("PRAGMA table_info(daily_bars)")).fetchall()
            print("   SQLite PRAGMA table_info result:")
            for row in schema_result:
                print(f"     {row[1]} ({row[2]})")
            
            actual_columns = {row[1] for row in schema_result}
            print(f"   Manually extracted columns: {sorted(list(actual_columns))}")
            print(f"   adj_close in manual extraction: {'adj_close' in actual_columns}")
            
        except Exception as e:
            print(f"   Error in manual verification: {e}")
            
        # Test 4: Clear cache and re-detect
        print("\n4. Testing cache refresh:")
        clear_table_columns_cache()
        print(f"   Cache cleared again: {_table_columns_cache}")
        
        columns_2 = _get_table_columns(conn, DailyBar)
        print(f"   Re-detected columns: {sorted(list(columns_2)) if columns_2 else None}")
        print(f"   Cache after re-detection: {_table_columns_cache}")
        
        # Test 5: DataFrame columns vs detected columns
        print("\n5. Testing DataFrame vs detected columns:")
        test_data = pd.DataFrame({
            'symbol': ['TEST2'],
            'ts': [date(2025, 9, 1)],
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [102.0],
            'adj_close': [101.5],
            'volume': [1000000],
            'vwap': [101.8],
            'trade_count': [1500]
        })
        
        df_columns = set(test_data.columns)
        detected_columns = columns_2 if columns_2 else set()
        
        print(f"   DataFrame columns: {sorted(list(df_columns))}")
        print(f"   Detected columns: {sorted(list(detected_columns))}")
        print(f"   Intersection: {sorted(list(df_columns.intersection(detected_columns)))}")
        print(f"   Missing in table: {sorted(list(df_columns - detected_columns))}")
        print(f"   Missing in DataFrame: {sorted(list(detected_columns - df_columns))}")

if __name__ == "__main__":
    test_column_detection()