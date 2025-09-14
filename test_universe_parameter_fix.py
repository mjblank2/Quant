#!/usr/bin/env python3
"""
Test for universe parameter limit fix.

This test validates that the universe rebuild function can handle
large numbers of symbols without hitting parameter limits.
"""

import os
import tempfile
import pandas as pd
import logging
from unittest.mock import patch
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def test_universe_rebuild_parameter_safety():
    """Test universe rebuild with various data sizes."""
    print("🧪 Testing universe rebuild parameter safety...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
        
        # Import after setting DATABASE_URL
        import db
        from data.universe import rebuild_universe
        
        # Create database tables
        db.Base.metadata.create_all(db.engine)
        
        # Test 1: Medium dataset (1000 symbols)
        medium_symbol_set = [
            {"symbol": f"MED{i:04d}", "name": f"Medium Company {i}"} 
            for i in range(1000)
        ]
        
        with patch('data.universe._list_small_cap_symbols', return_value=medium_symbol_set):
            result = rebuild_universe()
            
        print(f"✅ Successfully rebuilt universe with {len(result)} symbols")
        
        # Verify data was inserted
        with db.engine.connect() as conn:
            count_result = conn.execute(db.text("SELECT COUNT(*) FROM universe")).scalar()
            print(f"✅ Database contains {count_result} universe records")
            
        # Test 2: Large dataset (2000 symbols) - should still work with batching
        large_symbol_set = [
            {"symbol": f"LRG{i:04d}", "name": f"Large Company {i}"} 
            for i in range(2000)
        ]
        
        with patch('data.universe._list_small_cap_symbols', return_value=large_symbol_set):
            result = rebuild_universe()
            
        print(f"✅ Successfully rebuilt universe with {len(result)} symbols (large dataset)")
        
        # Verify updated data
        with db.engine.connect() as conn:
            count_result = conn.execute(db.text("SELECT COUNT(*) FROM universe")).scalar()
            print(f"✅ Database now contains {count_result} universe records")
            
        return True
        
    except Exception as e:
        print(f"❌ Universe rebuild test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        try:
            os.unlink(db_path)
        except:
            pass

def test_parameter_limit_logic():
    """Test the parameter limit logic directly."""
    print("🧪 Testing parameter limit calculations...")
    
    try:
        import db
        
        # Test parameter limit calculation for SQLite
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp:
            os.environ['DATABASE_URL'] = f'sqlite:///{tmp.name}'
            
            # Recreate engine with new URL
            from sqlalchemy import create_engine
            test_engine = create_engine(f'sqlite:///{tmp.name}')
            
            with test_engine.connect() as conn:
                max_params = db._max_bind_params_for_connection(conn)
                print(f"✅ SQLite parameter limit: {max_params}")
                
                # With 4 columns (symbol, name, included, last_updated), calculate safe batch size
                cols_per_row = 4
                safe_rows = max_params // cols_per_row
                print(f"✅ Safe rows per batch: {safe_rows}")
                
                if max_params == 999 and safe_rows == 249:
                    print("✅ Parameter calculations are correct")
                    return True
                else:
                    print(f"❌ Unexpected parameter calculations: max={max_params}, rows={safe_rows}")
                    return False
                    
    except Exception as e:
        print(f"❌ Parameter limit test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting universe parameter limit fix tests...")
    
    test1_success = test_universe_rebuild_parameter_safety()
    test2_success = test_parameter_limit_logic()
    
    if test1_success and test2_success:
        print("✅ All universe parameter fix tests passed!")
        exit(0)
    else:
        print("❌ Some universe parameter fix tests failed!")
        exit(1)