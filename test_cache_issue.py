#!/usr/bin/env python3
"""
Test to reproduce the cache issue that might cause adj_close warnings.
"""

import pandas as pd
import logging
from datetime import date
from db import DailyBar, upsert_dataframe, _get_table_columns, _table_columns_cache, engine, clear_table_columns_cache
from sqlalchemy import text
from unittest.mock import patch

# Set up logging to see warnings
logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')

def test_cache_stale_scenario():
    """Test scenario where cache becomes stale and causes column dropping warnings."""
    
    print("=" * 60)
    print("TESTING CACHE STALE SCENARIO")
    print("=" * 60)
    
    # Step 1: Populate cache with missing adj_close (simulate old schema)
    print("\n1. Simulating stale cache (without adj_close):")
    clear_table_columns_cache()
    
    # Manually populate cache with old schema (missing adj_close)
    _table_columns_cache['daily_bars'] = {
        'symbol', 'ts', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count'
        # Note: adj_close is intentionally missing to simulate old cache
    }
    print(f"   Stale cache set: {sorted(list(_table_columns_cache['daily_bars']))}")
    
    # Step 2: Try to insert data with adj_close
    print("\n2. Attempting upsert with adj_close in data:")
    test_data = pd.DataFrame({
        'symbol': ['CACHE_TEST'],
        'ts': [date(2025, 9, 1)],
        'open': [100.0],
        'high': [105.0],
        'low': [99.0],
        'close': [102.0],
        'adj_close': [101.5],  # This should trigger the warning
        'volume': [1000000],
        'vwap': [101.8],
        'trade_count': [1500]
    })
    
    print(f"   Data columns: {sorted(list(test_data.columns))}")
    print(f"   Cache columns: {sorted(list(_table_columns_cache['daily_bars']))}")
    print(f"   Missing in cache: {set(test_data.columns) - _table_columns_cache['daily_bars']}")
    
    # This should trigger the warning and cache refresh
    print("\n3. Running upsert (should show warning then refresh cache):")
    upsert_dataframe(test_data, DailyBar, ['symbol', 'ts'])
    
    print(f"   Cache after upsert: {sorted(list(_table_columns_cache.get('daily_bars', [])))}")

def test_mock_postgres_schema_issue():
    """Test scenario where PostgreSQL schema query fails."""
    
    print("\n" + "=" * 60)
    print("TESTING POSTGRESQL SCHEMA DETECTION FAILURE")
    print("=" * 60)
    
    clear_table_columns_cache()
    
    # Mock the PostgreSQL detection to return empty results (simulate schema issue)
    def mock_get_table_columns(connection, table):
        table_name = table.__tablename__
        
        # Simulate PostgreSQL that can't find columns in 'public' schema
        if "postgresql" in str(connection.engine.url).lower():
            # Simulate query returning no results
            print(f"   Mocked PostgreSQL: No columns found for {table_name} in 'public' schema")
            return set()  # Empty set - no columns found
        else:
            # For SQLite, use real implementation
            try:
                actual_columns_result = connection.execute(text(f"PRAGMA table_info({table_name})"))
                actual_columns = {row[1] for row in actual_columns_result.fetchall()}
                return actual_columns
            except Exception as e:
                print(f"   SQLite error: {e}")
                return None
    
    # Test with mocked failure
    print("\n1. Testing with mocked PostgreSQL schema detection failure:")
    with patch('db._get_table_columns', side_effect=mock_get_table_columns):
        test_data = pd.DataFrame({
            'symbol': ['MOCK_TEST'],
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
        
        print(f"   Data columns: {sorted(list(test_data.columns))}")
        print("\n2. Running upsert with mocked failure (should show multiple warnings):")
        upsert_dataframe(test_data, DailyBar, ['symbol', 'ts'])

if __name__ == "__main__":
    test_cache_stale_scenario()
    test_mock_postgres_schema_issue()