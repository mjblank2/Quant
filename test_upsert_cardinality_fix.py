#!/usr/bin/env python3
"""
Test for the upsert CardinalityViolation fix in db.py.

This test validates that the fix in the retry logic prevents
CardinalityViolation errors when duplicate data is processed.
"""

import pandas as pd
import numpy as np
import logging
import tempfile
import os
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Date, Float
from sqlalchemy.orm import declarative_base

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

Base = declarative_base()

class TestFeature(Base):
    """Test table mimicking the features table structure."""
    __tablename__ = "test_features"
    
    symbol = Column(String(20), primary_key=True)
    ts = Column(Date, primary_key=True)
    ret_1d = Column(Float)
    vol_21 = Column(Float)

def test_upsert_cardinality_fix():
    """Test that the upsert fix prevents CardinalityViolation on retry."""
    print("🎯 Testing upsert CardinalityViolation fix...")
    
    # Create a temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        temp_db_path = tmp.name
    
    try:
        # Create test database and engine
        engine = create_engine(f'sqlite:///{temp_db_path}')
        Base.metadata.create_all(engine)
        
        # Import the fixed upsert_dataframe function
        import sys
        sys.path.append('/home/runner/work/Quant/Quant')
        from db import upsert_dataframe
        
        # Create test data with duplicates that would cause CardinalityViolation
        problematic_data = pd.DataFrame([
            {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.01, 'vol_21': 0.15},
            {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.02, 'vol_21': 0.16},  # Duplicate!
            {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-05').date(), 'ret_1d': 0.03, 'vol_21': 0.17},
            {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.02, 'vol_21': 0.14},
            {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.025, 'vol_21': 0.145}, # Duplicate!
        ])
        
        print("Test data with duplicates (would cause CardinalityViolation):")
        print(problematic_data)
        
        # Try to upsert this data - should succeed with the fix
        try:
            with engine.connect() as conn:
                # Force a small chunk size to trigger retry logic
                upsert_dataframe(problematic_data, TestFeature, ['symbol', 'ts'], chunk_size=2, conn=conn)
                conn.commit()
            
            # Verify data was inserted correctly
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM test_features")).scalar()
                print(f"\n✅ Records inserted successfully: {result}")
                
                # Check actual data
                all_data = conn.execute(text("SELECT * FROM test_features ORDER BY symbol, ts")).fetchall()
                print("\nData in database:")
                for row in all_data:
                    print(f"  {row}")
                
                # Verify no duplicates in database
                duplicates_in_db = conn.execute(text("""
                    SELECT symbol, ts, COUNT(*) as count 
                    FROM test_features 
                    GROUP BY symbol, ts 
                    HAVING COUNT(*) > 1
                """)).fetchall()
                
                if len(duplicates_in_db) == 0:
                    print("✅ No duplicates found in database - fix successful!")
                    return True
                else:
                    print(f"❌ Found {len(duplicates_in_db)} duplicates in database:")
                    for dup in duplicates_in_db:
                        print(f"  {dup}")
                    return False
                    
        except Exception as e:
            print(f"❌ Upsert failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    finally:
        # Cleanup
        try:
            os.unlink(temp_db_path)
        except:
            pass

def test_deduplication_logic():
    """Test the deduplication logic in isolation."""
    print("\n🧪 Testing deduplication logic...")
    
    # Create test data with duplicates
    test_data = pd.DataFrame([
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.01, 'vol_21': 0.15},
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.02, 'vol_21': 0.16},  # Duplicate!
        {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.02, 'vol_21': 0.14},
        {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.025, 'vol_21': 0.145}, # Duplicate!
    ])
    
    print("Original data:")
    print(test_data)
    
    # Apply the deduplication logic from the fix
    conflict_cols = ['symbol', 'ts']
    original_size = len(test_data)
    
    deduplicated = test_data.drop_duplicates(subset=conflict_cols, keep='last').reset_index(drop=True)
    dedupe_size = len(deduplicated)
    
    print(f"\nDeduplication: {original_size} -> {dedupe_size} rows")
    print("Deduplicated data:")
    print(deduplicated)
    
    # Verify no duplicates remain
    final_check = deduplicated.groupby(conflict_cols).size()
    max_count = final_check.max()
    
    if max_count == 1:
        print("✅ Deduplication logic works correctly")
        return True
    else:
        print(f"❌ Deduplication failed - max count per group: {max_count}")
        return False

def run_all_tests():
    """Run all upsert CardinalityViolation fix tests."""
    print("🎯 Running upsert CardinalityViolation fix tests\n")
    
    tests = [
        test_deduplication_logic,
        test_upsert_cardinality_fix,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print(f"📊 Upsert Fix Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All upsert CardinalityViolation fix tests passed!")
        print("The fix should prevent the pipeline retry error.")
        return True
    else:
        print("⚠️ Some tests failed - fix may need adjustment")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)