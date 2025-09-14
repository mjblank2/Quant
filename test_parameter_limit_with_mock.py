#!/usr/bin/env python3
"""
Test parameter limit handling by mocking a real parameter limit error.
This validates that the retry logic works correctly without needing to actually hit database limits.
"""

import pandas as pd
import numpy as np
import logging
import tempfile
import os
from datetime import date, datetime
from unittest.mock import patch, Mock
from sqlalchemy import create_engine, text

# Setup detailed logging to capture all levels
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')
log = logging.getLogger(__name__)

def test_parameter_limit_retry_logic():
    """Test that parameter limit errors trigger proper retry logic."""
    
    # Set up test database
    db_file = tempfile.mktemp(suffix='.db')
    test_db_url = f"sqlite:///{db_file}"
    
    print(f"📊 Testing parameter limit retry logic with {test_db_url}")
    
    try:
        # Set environment for db.py
        os.environ['DATABASE_URL'] = test_db_url
        
        # Import after setting DATABASE_URL
        import db
        
        # Create tables
        db.create_tables()
        
        # Create test data
        data = []
        for i in range(50):
            record = {
                'symbol': f'SYMB{i:04d}',
                'name': f'Company Name {i}',  
                'exchange': 'NASDAQ',
                'market_cap': 1000000.0 + i * 1000,
                'adv_usd_20': 50000.0 + i * 100,
                'included': True,
                'last_updated': datetime.now()
            }
            data.append(record)
        
        test_df = pd.DataFrame(data)
        print(f"Created test DataFrame with {len(test_df)} rows and {len(test_df.columns)} columns")
        
        conflict_cols = ['symbol']
        
        # Mock the connection.execute method to raise a parameter limit error on first call
        call_count = 0
        original_execute = None
        
        def mock_execute(stmt, *args, **kwargs):
            nonlocal call_count, original_execute
            call_count += 1
            
            if call_count == 1:
                # First call: simulate parameter limit error
                from sqlalchemy.exc import OperationalError
                # Simulate actual SQLite parameter limit error
                raise OperationalError("too many SQL variables", {}, None)
            else:
                # Subsequent calls: use original execute method
                return original_execute(stmt, *args, **kwargs)
        
        # Patch the connection execute method during upsert
        with patch.object(db.engine, 'begin') as mock_begin:
            # Create a real connection for the test
            real_connection = db.engine.connect()
            original_execute = real_connection.execute
            
            # Mock the context manager to return our controlled connection
            mock_context = Mock()
            mock_context.__enter__ = Mock(return_value=real_connection)
            mock_context.__exit__ = Mock(return_value=None)
            mock_begin.return_value = mock_context
            
            # Replace execute with our mock
            real_connection.execute = mock_execute
            
            try:
                # This should trigger the parameter limit error and retry
                db.upsert_dataframe(test_df, db.Universe, conflict_cols, chunk_size=50000)
                print("✅ Upsert completed successfully with retry logic")
                
                # Verify that retry was triggered (should have been called more than once)
                if call_count > 1:
                    print(f"✅ Retry logic was triggered: {call_count} execute calls made")
                else:
                    print(f"❌ Retry logic was not triggered: only {call_count} execute call made")
                    return False
                    
            except Exception as e:
                print(f"❌ Upsert failed: {e}")
                return False
            finally:
                real_connection.close()
        
        print("✅ Parameter limit retry test passed")
        return True
        
    finally:
        # Clean up
        try:
            if os.path.exists(db_file):
                os.unlink(db_file)
        except:
            pass

def test_conservative_parameter_limits():
    """Test that the parameter limit calculation is conservative enough."""
    
    # Set up test database  
    db_file = tempfile.mktemp(suffix='.db')
    test_db_url = f"sqlite:///{db_file}"
    
    print(f"📊 Testing conservative parameter limit calculation with {test_db_url}")
    
    try:
        # Set environment for db.py
        os.environ['DATABASE_URL'] = test_db_url
        
        # Import after setting DATABASE_URL
        import db
        
        # Create tables
        db.create_tables()
        
        # Test parameter limit calculation
        with db.engine.connect() as conn:
            max_params = db._max_bind_params_for_connection(conn)
            print(f"📊 Max parameters for SQLite connection: {max_params}")
            
            # For SQLite, this should be 999 (conservative limit)
            if max_params != 999:
                print(f"❌ Expected 999 parameters for SQLite, got {max_params}")
                return False
                
            # Test with various column counts
            test_cases = [
                (7, 142),   # Universe table: 999 / 7 = 142 rows
                (23, 43),   # Feature table: 999 / 23 = 43 rows  
                (10, 99),   # 10 columns: 999 / 10 = 99 rows
                (50, 19),   # 50 columns: 999 / 50 = 19 rows
            ]
            
            for col_count, expected_max_rows in test_cases:
                # Simulate the calculation from upsert_dataframe
                theoretical_max_rows = max_params // max(1, col_count)
                per_stmt_rows = max(1, min(50000, theoretical_max_rows, 1000))  # Same logic as in upsert_dataframe
                
                print(f"📊 {col_count} columns: max {theoretical_max_rows} rows per statement (capped at {per_stmt_rows})")
                
                if theoretical_max_rows != expected_max_rows:
                    print(f"❌ Expected {expected_max_rows} max rows for {col_count} columns, got {theoretical_max_rows}")
                    return False
                    
        print("✅ Conservative parameter limit calculation test passed")
        return True
        
    finally:
        # Clean up
        try:
            if os.path.exists(db_file):
                os.unlink(db_file)
        except:
            pass

if __name__ == "__main__":
    print("🚀 Starting parameter limit tests with mocking...")
    
    success1 = test_parameter_limit_retry_logic()
    success2 = test_conservative_parameter_limits()
    
    if success1 and success2:
        print("\n✅ All parameter limit tests passed")
    else:
        print("\n❌ Some parameter limit tests failed")
        exit(1)