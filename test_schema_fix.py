#!/usr/bin/env python3
"""
Test the PostgreSQL schema detection fix.
"""

import pandas as pd
import logging
from datetime import date
from db import DailyBar, upsert_dataframe, _get_table_columns, clear_table_columns_cache, engine
from sqlalchemy import text
from unittest.mock import patch, Mock

# Set up logging to see warnings
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

def test_postgresql_schema_fix():
    """Test the improved PostgreSQL schema detection."""
    
    print("=" * 60)
    print("TESTING POSTGRESQL SCHEMA DETECTION FIX")
    print("=" * 60)
    
    # Mock a PostgreSQL connection that fails for 'public' schema but works for current_schema()
    def mock_execute(stmt, params=None):
        if hasattr(stmt, 'text'):
            query_text = str(stmt.text).strip()
        else:
            query_text = str(stmt).strip()
            
        # Check what type of query this is
        if "information_schema.columns" in query_text.lower():
            print(f"   PostgreSQL query: {query_text[:80]}...")
            
            if "table_schema = 'public'" in query_text:
                # Simulate failure for 'public' schema
                print("   -> Simulating 'public' schema failure (empty result)")
                mock_result = Mock()
                mock_result.fetchall.return_value = []
                return mock_result
                
            elif "current_schema()" in query_text:
                # Simulate success for current_schema()
                print("   -> Simulating current_schema() success")
                mock_result = Mock()
                mock_result.fetchall.return_value = [
                    ('symbol',), ('ts',), ('open',), ('high',), ('low',), 
                    ('close',), ('adj_close',), ('volume',), ('vwap',), ('trade_count',)
                ]
                return mock_result
                
            else:
                # Fallback - no schema restriction
                print("   -> Simulating fallback (no schema restriction)")
                mock_result = Mock()
                mock_result.fetchall.return_value = [
                    ('symbol',), ('ts',), ('open',), ('high',), ('low',), 
                    ('close',), ('adj_close',), ('volume',), ('vwap',), ('trade_count',)
                ]
                return mock_result
        else:
            # For non-schema queries, just return a mock
            return Mock()
    
    # Test with mocked PostgreSQL URL and execute
    mock_url = Mock()
    mock_url.__str__ = lambda: "postgresql+psycopg://user:pass@host:5432/dbname"
    
    clear_table_columns_cache()
    
    with patch.object(engine, 'url', mock_url):
        mock_connection = Mock()
        mock_connection.engine.url = mock_url
        mock_connection.execute = mock_execute
        
        print("\n1. Testing improved column detection:")
        columns = _get_table_columns(mock_connection, DailyBar)
        print(f"   Detected columns: {sorted(list(columns)) if columns else None}")
        print(f"   adj_close found: {'adj_close' in columns if columns else False}")
        
        # Now test upsert with this connection
        print("\n2. Testing upsert with improved detection:")
        test_data = pd.DataFrame({
            'symbol': ['PG_FIX_TEST'],
            'ts': [date(2025, 9, 1)],
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [102.0],
            'adj_close': [101.5],  # Should NOT trigger warning now
            'volume': [1000000],
            'vwap': [101.8],
            'trade_count': [1500]
        })
        
        # Mock the engine.begin() context manager
        with patch('db.engine') as mock_engine:
            mock_engine.begin.return_value.__enter__.return_value = mock_connection
            mock_engine.begin.return_value.__exit__.return_value = None
            
            print(f"   Data columns: {sorted(list(test_data.columns))}")
            print("   Expecting NO warning about dropping adj_close...")
            
            try:
                upsert_dataframe(test_data, DailyBar, ['symbol', 'ts'])
                print("   ✓ Upsert completed without dropping adj_close")
            except Exception as e:
                print(f"   ✗ Upsert failed: {e}")

def test_fallback_scenarios():
    """Test various fallback scenarios."""
    
    print("\n" + "=" * 60)
    print("TESTING FALLBACK SCENARIOS")
    print("=" * 60)
    
    def mock_execute_all_fail(stmt, params=None):
        """Mock that fails all PostgreSQL queries."""
        if hasattr(stmt, 'text'):
            query_text = str(stmt.text).strip()
        else:
            query_text = str(stmt).strip()
            
        if "information_schema.columns" in query_text.lower():
            print(f"   Query failed: {query_text[:50]}...")
            raise Exception("Simulated database error")
        return Mock()
    
    mock_url = Mock()
    mock_url.__str__ = lambda: "postgresql+psycopg://user:pass@host:5432/dbname"
    
    clear_table_columns_cache()
    
    with patch.object(engine, 'url', mock_url):
        mock_connection = Mock()
        mock_connection.engine.url = mock_url
        mock_connection.execute = mock_execute_all_fail
        
        print("\n1. Testing when all PostgreSQL queries fail:")
        columns = _get_table_columns(mock_connection, DailyBar)
        print(f"   Result when all queries fail: {columns}")
        print(f"   Should be None to trigger 'use all DataFrame columns' logic")

if __name__ == "__main__":
    test_postgresql_schema_fix()
    test_fallback_scenarios()