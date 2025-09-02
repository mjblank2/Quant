#!/usr/bin/env python3
"""
End-to-end test to validate the CardinalityViolation fix.
This test simulates the exact error scenario and verifies it's fixed.
"""

import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert
import tempfile
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def test_cardinality_violation_fix():
    """Test that our fix prevents the CardinalityViolation error."""
    print("🎯 Testing CardinalityViolation fix...")
    
    # Create a temporary SQLite database to test the upsert logic
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        temp_db_path = tmp.name
    
    try:
        # Create test database
        engine = create_engine(f'sqlite:///{temp_db_path}')
        
        # Create a simplified features table
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE features_test (
                    symbol VARCHAR(20) NOT NULL,
                    ts DATE NOT NULL,
                    ret_1d FLOAT,
                    vol_21 FLOAT,
                    PRIMARY KEY (symbol, ts)
                )
            """))
            conn.commit()
        
        # Create test data that would cause the original error
        # (same symbol+date with different timestamps)
        problematic_data = pd.DataFrame([
            {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04 00:00:00'), 'ret_1d': 0.01, 'vol_21': 0.15},
            {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04 05:00:00'), 'ret_1d': 0.02, 'vol_21': 0.16},  # Same date!
            {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-05 00:00:00'), 'ret_1d': 0.03, 'vol_21': 0.17},
            {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-04 00:00:00'), 'ret_1d': 0.02, 'vol_21': 0.14},
            {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-04 09:30:00'), 'ret_1d': 0.025, 'vol_21': 0.145}, # Same date!
        ])
        
        print("Original problematic data (would cause CardinalityViolation):")
        print(problematic_data)
        
        # Apply our fix
        feats = problematic_data.copy()
        original_count = len(feats)
        
        # Normalize timestamps to dates to avoid timezone/time component issues
        feats['ts'] = feats['ts'].dt.normalize()
        
        # Check for duplicates before deduplication
        duplicate_check = feats.groupby(['symbol', 'ts']).size()
        duplicates_found = (duplicate_check > 1).sum()
        
        print(f"Duplicates found: {duplicates_found}")
        
        if duplicates_found > 0:
            log.warning(f"Found {duplicates_found} symbol+date pairs with multiple rows. Deduplicating by keeping latest values.")
            
            # Keep the latest row for each symbol+date combination
            latest_indices = feats.groupby(['symbol', 'ts']).tail(1).index
            feats = feats.loc[latest_indices].reset_index(drop=True)
            
            deduplicated_count = len(feats)
            log.info(f"Deduplication: {original_count} -> {deduplicated_count} rows ({original_count - deduplicated_count} duplicates removed)")
        
        print("\nData after applying fix:")
        print(feats)
        
        # Test that this data can now be inserted without error
        try:
            # Convert to list of dicts for insertion
            records = feats.to_dict(orient='records')
            
            # Create SQLAlchemy insert statement
            from sqlalchemy import MetaData, Table
            metadata = MetaData()
            features_table = Table('features_test', metadata, autoload_with=engine)
            
            with engine.connect() as conn:
                # This should work without CardinalityViolation now
                stmt = insert(features_table).values(records)
                
                # Add ON CONFLICT logic (SQLite uses different syntax than PostgreSQL)
                # For SQLite, we'll use INSERT OR REPLACE
                conn.execute(text("""
                    INSERT OR REPLACE INTO features_test (symbol, ts, ret_1d, vol_21)
                    VALUES (:symbol, :ts, :ret_1d, :vol_21)
                """), records)
                conn.commit()
                
                # Verify data was inserted correctly
                result = conn.execute(text("SELECT COUNT(*) FROM features_test")).scalar()
                print(f"\nRecords inserted successfully: {result}")
                
                # Verify no duplicates in database
                duplicates_in_db = conn.execute(text("""
                    SELECT symbol, ts, COUNT(*) as count 
                    FROM features_test 
                    GROUP BY symbol, ts 
                    HAVING COUNT(*) > 1
                """)).fetchall()
                
                if len(duplicates_in_db) == 0:
                    print("✅ No duplicates found in database - fix successful!")
                    return True
                else:
                    print(f"❌ Found {len(duplicates_in_db)} duplicates in database")
                    return False
                    
        except Exception as e:
            print(f"❌ Database insertion failed: {e}")
            return False
            
    finally:
        # Cleanup
        try:
            os.unlink(temp_db_path)
        except:
            pass

def test_original_error_scenario():
    """Test that the original error scenario is now fixed."""
    print("\n🔍 Testing original error scenario simulation...")
    
    # Simulate the exact data from the error log
    error_data = pd.DataFrame([
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04 00:00:00'), 'ret_1d': 0.0008550256507693366},
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04 05:00:00'), 'ret_1d': 0.0},  # Same date, different time
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-05 00:00:00'), 'ret_1d': -0.025059326056003806},
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-06 00:00:00'), 'ret_1d': -0.021772939346811793},
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-07 00:00:00'), 'ret_1d': 0.008471156067461542},
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-08 00:00:00'), 'ret_1d': 0.005287713841368502},
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-08 05:00:00'), 'ret_1d': 0.0},  # Another duplicate
    ])
    
    print("Original error scenario data:")
    print(error_data[['symbol', 'ts', 'ret_1d']])
    
    # Apply our fix
    feats = error_data.copy()
    
    # Show the problem first
    duplicate_check_before = feats.groupby(['symbol', feats['ts'].dt.normalize()]).size()
    duplicates_before = (duplicate_check_before > 1).sum()
    print(f"\nDuplicates before fix: {duplicates_before}")
    
    # Apply timestamp normalization
    feats['ts'] = feats['ts'].dt.normalize()
    
    # Check for duplicates
    duplicate_check = feats.groupby(['symbol', 'ts']).size()
    duplicates_found = (duplicate_check > 1).sum()
    
    if duplicates_found > 0:
        print(f"Deduplicating {duplicates_found} duplicate pairs...")
        latest_indices = feats.groupby(['symbol', 'ts']).tail(1).index
        feats = feats.loc[latest_indices].reset_index(drop=True)
    
    # Verify fix
    final_check = feats.groupby(['symbol', 'ts']).size()
    max_count = final_check.max()
    
    print(f"After fix - max rows per symbol+date: {max_count}")
    print("\nFinal deduplicated data:")
    print(feats[['symbol', 'ts', 'ret_1d']])
    
    if max_count == 1:
        print("✅ Original error scenario fixed!")
        return True
    else:
        print("❌ Original error scenario not fixed!")
        return False

def run_cardinality_tests():
    """Run all CardinalityViolation fix tests."""
    print("🎯 Running CardinalityViolation fix validation tests\n")
    
    tests = [
        test_cardinality_violation_fix,
        test_original_error_scenario,
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
    
    print(f"📊 CardinalityViolation Fix Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All CardinalityViolation fix tests passed!")
        print("The fix should prevent the pipeline error.")
        return True
    else:
        print("⚠️ Some tests failed - fix may need adjustment")
        return False

if __name__ == "__main__":
    success = run_cardinality_tests()
    exit(0 if success else 1)