#!/usr/bin/env python3
"""
Debug test to reproduce the exact CardinalityViolation issue from the error log.

This test simulates the exact conditions that cause the error in production.
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
    ret_5d = Column(Float)
    vol_21 = Column(Float)

def test_cardinality_reproduction():
    """Reproduce the exact CardinalityViolation issue."""
    print("🔍 Reproducing CardinalityViolation scenario...")
    
    # Create a temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        temp_db_path = tmp.name
    
    try:
        # Create test database and engine
        engine = create_engine(f'sqlite:///{temp_db_path}')
        Base.metadata.create_all(engine)
        
        # Import the upsert_dataframe function with DATABASE_URL set
        os.environ['DATABASE_URL'] = f'sqlite:///{temp_db_path}'
        import sys
        sys.path.append('/home/runner/work/Quant/Quant')
        from db import upsert_dataframe
        
        # Create test data that simulates the actual feature engineering output
        # This simulates what happens when multiple symbols with the same timestamp are processed
        test_data = []
        
        # Simulate feature engineering creating duplicate entries
        base_date = pd.Timestamp('2016-01-04').date()
        
        # Symbol AAPL with same timestamp multiple times (this could happen during processing)
        test_data.extend([
            {'symbol': 'AAPL', 'ts': base_date, 'ret_1d': 0.01, 'ret_5d': 0.05, 'vol_21': 0.15},
            {'symbol': 'AAPL', 'ts': base_date, 'ret_1d': 0.012, 'ret_5d': 0.052, 'vol_21': 0.151},  # Duplicate same symbol+ts!
        ])
        
        # Symbol MSFT with same timestamp multiple times  
        test_data.extend([
            {'symbol': 'MSFT', 'ts': base_date, 'ret_1d': 0.02, 'ret_5d': 0.04, 'vol_21': 0.14},
            {'symbol': 'MSFT', 'ts': base_date, 'ret_1d': 0.021, 'ret_5d': 0.041, 'vol_21': 0.141},  # Duplicate same symbol+ts!
        ])
        
        problematic_df = pd.DataFrame(test_data)
        
        print("Test data with duplicates (should cause CardinalityViolation):")
        print(problematic_df)
        print()
        
        # Check duplicates before upsert
        duplicate_check = problematic_df.groupby(['symbol', 'ts']).size()
        duplicates_found = (duplicate_check > 1).sum()
        print(f"Duplicates found in test data: {duplicates_found}")
        if duplicates_found > 0:
            print("Duplicate groups:")
            print(duplicate_check[duplicate_check > 1])
        print()
        
        # Try to upsert this data - this should trigger the CardinalityViolation fix
        try:
            with engine.connect() as conn:
                # Use a small chunk size to test the batching logic
                upsert_dataframe(problematic_df, TestFeature, ['symbol', 'ts'], chunk_size=2, conn=conn)
                conn.commit()
            
            # Verify data was inserted correctly
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM test_features")).scalar()
                print(f"✅ Records inserted successfully: {result}")
                
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
                    print("✅ No duplicates found in database - fix works!")
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

def test_batch_processing_simulation():
    """Test simulating how features.py processes batches."""
    print("🔬 Testing batch processing simulation...")
    
    # Simulate the feature engineering process that creates the DataFrame
    # This mimics the logic from models/features.py lines 218-220
    
    out_frames = []
    
    # Simulate processing symbol AAPL
    aapl_data = pd.DataFrame([
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.01, 'ret_5d': 0.05, 'vol_21': 0.15},
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-05').date(), 'ret_1d': 0.02, 'ret_5d': 0.06, 'vol_21': 0.16},
    ])
    out_frames.append(aapl_data)
    
    # Simulate processing symbol MSFT
    msft_data = pd.DataFrame([
        {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.015, 'ret_5d': 0.045, 'vol_21': 0.14},
        {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-05').date(), 'ret_1d': 0.025, 'ret_5d': 0.055, 'vol_21': 0.15},
    ])
    out_frames.append(msft_data)
    
    # This is the critical line from features.py:219
    feats = pd.concat(out_frames, ignore_index=True)
    
    print("Simulated feats DataFrame:")
    print(feats)
    print()
    
    # Check for duplicates (there shouldn't be any in this normal case)
    duplicate_check = feats.groupby(['symbol', 'ts']).size()
    duplicates_found = (duplicate_check > 1).sum()
    print(f"Duplicates in normal batch processing: {duplicates_found}")
    
    if duplicates_found == 0:
        print("✅ Normal batch processing doesn't create duplicates")
        return True
    else:
        print("❌ Normal batch processing created duplicates!")
        print(duplicate_check[duplicate_check > 1])
        return False

def test_timestamp_processing_edge_case():
    """Test edge cases in timestamp processing that might create duplicates."""
    print("⚡ Testing timestamp processing edge cases...")
    
    # Simulate problematic timestamp processing
    # This could happen if the same symbol's data gets processed multiple times
    # or if there are timezone/normalization issues
    
    test_scenarios = []
    
    # Scenario 1: Same symbol processed twice (could happen in parallel processing)
    scenario1 = pd.DataFrame([
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.01, 'vol_21': 0.15},
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.011, 'vol_21': 0.151},  # Slight difference
    ])
    test_scenarios.append(("Symbol processed twice", scenario1))
    
    # Scenario 2: Timestamp with different time components that normalize to same date
    base_ts = pd.Timestamp('2016-01-04')
    scenario2 = pd.DataFrame([
        {'symbol': 'AAPL', 'ts': base_ts.date(), 'ret_1d': 0.01, 'vol_21': 0.15},
        {'symbol': 'AAPL', 'ts': (base_ts + pd.Timedelta(hours=6)).date(), 'ret_1d': 0.011, 'vol_21': 0.151},  # Same date!
    ])
    test_scenarios.append(("Timestamp normalization", scenario2))
    
    for scenario_name, scenario_data in test_scenarios:
        print(f"\nTesting scenario: {scenario_name}")
        print(scenario_data)
        
        duplicate_check = scenario_data.groupby(['symbol', 'ts']).size()
        duplicates_found = (duplicate_check > 1).sum()
        print(f"Duplicates found: {duplicates_found}")
        
        if duplicates_found > 0:
            print("❌ This scenario creates duplicates!")
            print(duplicate_check[duplicate_check > 1])
            return False
    
    print("✅ All timestamp scenarios handled correctly")
    return True

def run_debug_tests():
    """Run all debug tests to identify the CardinalityViolation issue."""
    print("🎯 Running CardinalityViolation debug tests\n")
    
    tests = [
        test_batch_processing_simulation,
        test_timestamp_processing_edge_case,
        test_cardinality_reproduction,
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
    
    print(f"📊 Debug Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All debug tests passed!")
        print("The CardinalityViolation fix should be working correctly.")
        return True
    else:
        print("⚠️ Some tests failed - this helps identify the issue")
        return False

if __name__ == "__main__":
    success = run_debug_tests()
    exit(0 if success else 1)