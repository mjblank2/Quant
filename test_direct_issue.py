#!/usr/bin/env python3
"""
Test to reproduce the exact adj_close warning scenario.
"""

import pandas as pd
import logging
from datetime import date
from db import DailyBar, upsert_dataframe, engine
from sqlalchemy import text
from unittest.mock import patch, Mock

# Set up logging to see warnings exactly as in production
logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')

def test_postgresql_url_mock():
    """Test with engine URL that looks like PostgreSQL but actually queries fail."""
    
    print("=" * 60)
    print("TESTING POSTGRESQL URL MOCK")
    print("=" * 60)
    
    test_data = pd.DataFrame({
        'symbol': ['PG_TEST'],
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
    
    # Mock engine.url to return a PostgreSQL-like URL
    mock_url = Mock()
    mock_url.__str__ = lambda: "postgresql+psycopg://user:pass@host:5432/dbname"
    
    def mock_execute(stmt, params=None):
        """Mock execute that fails for PostgreSQL information_schema query."""
        if hasattr(stmt, 'text'):
            query_text = str(stmt.text).strip()
        else:
            query_text = str(stmt).strip()
            
        # If it's the PostgreSQL information_schema query, return empty results
        if "information_schema.columns" in query_text.lower():
            print(f"   Mocking PostgreSQL query failure: {query_text[:50]}...")
            # Return empty result set
            mock_result = Mock()
            mock_result.fetchall.return_value = []  # No columns found
            return mock_result
        else:
            # For other queries (like the actual INSERT), use real execution
            return engine._execute_connection(stmt, params)
    
    # Patch both the engine URL and connection execute method
    with patch.object(engine, 'url', mock_url):
        with patch('db.engine') as mock_engine:
            # Set up the mock engine to use our mock execute for column detection
            mock_connection = Mock()
            mock_connection.engine.url = mock_url
            mock_connection.execute = mock_execute
            
            # Mock the engine.begin() context manager
            mock_engine.begin.return_value.__enter__.return_value = mock_connection
            mock_engine.begin.return_value.__exit__.return_value = None
            
            print("\n1. Running upsert with PostgreSQL URL and failing schema query:")
            print(f"   Data includes adj_close: {'adj_close' in test_data.columns}")
            print("   Expecting warning about dropping adj_close column...")
            
            try:
                upsert_dataframe(test_data, DailyBar, ['symbol', 'ts'])
                print("   Upsert completed")
            except Exception as e:
                print(f"   Upsert failed: {e}")

def test_direct_warning_trigger():
    """Test by directly simulating the problematic scenario."""
    
    print("\n" + "=" * 60)
    print("TESTING DIRECT WARNING TRIGGER")
    print("=" * 60)
    
    # Import and directly test the problematic logic
    from db import _get_table_columns, _table_columns_cache, clear_table_columns_cache
    
    clear_table_columns_cache()
    
    test_data = pd.DataFrame({
        'symbol': ['DIRECT_TEST'],
        'ts': [date(2025, 9, 1)],
        'adj_close': [101.5],  # Only adj_close to make the issue clear
        'close': [102.0]
    })
    
    # Manually trigger the exact scenario from upsert_dataframe
    with engine.begin() as connection:
        print("\n1. Testing column detection:")
        
        # First, get actual columns (this should work)
        actual_columns = _get_table_columns(connection, DailyBar)
        print(f"   Actual columns detected: {sorted(list(actual_columns)) if actual_columns else None}")
        
        # Now test the logic from upsert_dataframe
        df_columns = set(test_data.columns)
        print(f"   DataFrame columns: {sorted(list(df_columns))}")
        
        if actual_columns is None:
            print("\n2. Column detection failed - using all DataFrame columns")
            valid_columns = df_columns
        else:
            valid_columns = df_columns.intersection(actual_columns)
            print(f"   Valid columns (intersection): {sorted(list(valid_columns))}")
            
            if valid_columns != df_columns:
                missing_in_table = df_columns - actual_columns
                print(f"   Missing in table: {sorted(list(missing_in_table))}")
                
                # This is where the warning would be logged
                if missing_in_table:
                    print(f"   WARNING: Would drop columns not present in daily_bars: {missing_in_table}")

if __name__ == "__main__":
    test_postgresql_url_mock()
    test_direct_warning_trigger()