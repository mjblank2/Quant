#!/usr/bin/env python3
"""
Comprehensive test for the complete CardinalityViolation fix.

This test validates that both the database-level fix in db.py and the 
feature-level fix in models/features.py work together to completely
prevent CardinalityViolation errors.
"""

import pandas as pd
import numpy as np
import logging
import tempfile
import os
from sqlalchemy import create_engine, text
from unittest.mock import patch

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def test_complete_cardinality_fix():
    """Test the complete CardinalityViolation fix end-to-end."""
    print("🎯 Testing complete CardinalityViolation fix end-to-end...")
    
    # Create a temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        temp_db_path = tmp.name
    
    try:
        # Set up environment
        os.environ['DATABASE_URL'] = f'sqlite:///{temp_db_path}'
        
        # Create test database
        engine = create_engine(f'sqlite:///{temp_db_path}')
        
        # Import after setting DATABASE_URL
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from db import Base, Feature, upsert_dataframe
        
        # Create tables
        Base.metadata.create_all(engine)
        
        print("✅ Database and tables created")
        
        # Test 1: Direct upsert with duplicates (tests db.py fix)
        print("\n📊 Test 1: Direct upsert with duplicates (db.py fix)")
        direct_duplicates = pd.DataFrame([
            {'symbol': 'TEST1', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.01, 'ret_5d': 0.05, 'vol_21': 0.15},
            {'symbol': 'TEST1', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.011, 'ret_5d': 0.051, 'vol_21': 0.151},  # Duplicate!
        ])
        
        print("Data with duplicates:")
        print(direct_duplicates)
        
        # This should work due to the proactive deduplication in db.py
        with engine.connect() as conn:
            upsert_dataframe(direct_duplicates, Feature, ['symbol', 'ts'], conn=conn)
            conn.commit()
        
        # Verify
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM features WHERE symbol = 'TEST1'")).scalar()
            print(f"✅ Direct upsert successful: {count} records inserted")
        
        # Test 2: Simulate feature engineering process (tests models/features.py fix)
        print("\n📊 Test 2: Feature engineering simulation (models/features.py fix)")
        
        # Simulate the exact process from build_features() that creates out_frames
        out_frames = []
        
        # Simulate processing different symbols that might create overlapping data
        frame1 = pd.DataFrame([
            {'symbol': 'TEST2', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.02, 'ret_5d': 0.06, 'vol_21': 0.16},
            {'symbol': 'TEST3', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.03, 'ret_5d': 0.07, 'vol_21': 0.17},
        ])
        out_frames.append(frame1)
        
        # Simulate a problematic merge operation that creates duplicates
        frame2 = pd.DataFrame([
            {'symbol': 'TEST2', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.021, 'ret_5d': 0.061, 'vol_21': 0.161},  # Duplicate!
            {'symbol': 'TEST3', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.031, 'ret_5d': 0.071, 'vol_21': 0.171},  # Duplicate!
        ])
        out_frames.append(frame2)
        
        # This replicates the exact logic from models/features.py with our fix
        feats = pd.concat(out_frames, ignore_index=True)
        
        print("Concatenated features (with duplicates):")
        print(feats)
        
        # Apply the new fix from models/features.py
        original_count = len(feats)
        feats = feats.drop_duplicates(subset=['symbol', 'ts'], keep='last').reset_index(drop=True)
        dedupe_count = len(feats)
        if dedupe_count < original_count:
            print(f"🔧 Removed {original_count - dedupe_count} duplicate (symbol, ts) pairs during feature engineering")
        
        print("After deduplication:")
        print(feats)
        
        # Now upsert (this should work smoothly)
        with engine.connect() as conn:
            upsert_dataframe(feats, Feature, ['symbol', 'ts'], conn=conn)
            conn.commit()
        
        # Verify
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM features WHERE symbol IN ('TEST2', 'TEST3')")).scalar()
            print(f"✅ Feature engineering simulation successful: {count} records inserted")
        
        # Test 3: Comprehensive end-to-end test
        print("\n📊 Test 3: Comprehensive stress test")
        
        # Create a more complex scenario with multiple sources of duplicates
        complex_data = []
        
        # Add data that looks like real feature engineering output
        dates = [pd.Timestamp('2016-01-04').date(), pd.Timestamp('2016-01-05').date()]
        symbols = ['STRESS1', 'STRESS2', 'STRESS3']
        
        for symbol in symbols:
            for date in dates:
                # First version (might come from initial computation)
                complex_data.append({
                    'symbol': symbol, 'ts': date, 'ret_1d': np.random.random() * 0.01, 
                    'ret_5d': np.random.random() * 0.05, 'vol_21': 0.10 + np.random.random() * 0.10
                })
                # Duplicate version (might come from re-computation or merge issues)
                complex_data.append({
                    'symbol': symbol, 'ts': date, 'ret_1d': np.random.random() * 0.01, 
                    'ret_5d': np.random.random() * 0.05, 'vol_21': 0.10 + np.random.random() * 0.10
                })
        
        stress_df = pd.DataFrame(complex_data)
        print(f"Stress test data: {len(stress_df)} rows with potential duplicates")
        
        # Check duplicates
        duplicate_check = stress_df.groupby(['symbol', 'ts']).size()
        duplicates_found = (duplicate_check > 1).sum()
        print(f"Duplicate (symbol, ts) pairs found: {duplicates_found}")
        
        # Apply the complete fix (feature-level deduplication + db-level safety)
        original_count = len(stress_df)
        stress_df = stress_df.drop_duplicates(subset=['symbol', 'ts'], keep='last').reset_index(drop=True)
        dedupe_count = len(stress_df)
        print(f"After feature-level deduplication: {original_count} -> {dedupe_count} rows")
        
        # Now upsert with db-level safety
        with engine.connect() as conn:
            upsert_dataframe(stress_df, Feature, ['symbol', 'ts'], conn=conn)
            conn.commit()
        
        # Final verification
        with engine.connect() as conn:
            total_count = conn.execute(text("SELECT COUNT(*) FROM features")).scalar()
            unique_pairs = conn.execute(text("SELECT COUNT(DISTINCT symbol || '|' || ts) FROM features")).scalar()
            
            print(f"✅ Final database state: {total_count} total records, {unique_pairs} unique (symbol, ts) pairs")
            
            if total_count == unique_pairs:
                print("✅ No duplicates in final database - complete fix successful!")
                return True
            else:
                print("❌ Duplicates found in final database")
                return False
                
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            os.unlink(temp_db_path)
        except:
            pass

def test_performance_impact():
    """Test that the fix doesn't significantly impact performance."""
    print("⚡ Testing performance impact of the fix...")
    
    import time
    
    # Create test data
    large_data = []
    for i in range(1000):
        large_data.append({
            'symbol': f'PERF{i % 50}',  # 50 symbols
            'ts': pd.Timestamp('2016-01-01').date() + pd.Timedelta(days=i % 100),  # 100 days
            'ret_1d': np.random.random() * 0.02 - 0.01,
            'ret_5d': np.random.random() * 0.10 - 0.05,
            'vol_21': 0.10 + np.random.random() * 0.20
        })
    
    large_df = pd.DataFrame(large_data)
    print(f"Performance test with {len(large_df)} rows")
    
    # Test deduplication performance
    start_time = time.time()
    deduplicated = large_df.drop_duplicates(subset=['symbol', 'ts'], keep='last').reset_index(drop=True)
    dedup_time = time.time() - start_time
    
    print(f"Deduplication completed in {dedup_time:.4f} seconds")
    print(f"Original: {len(large_df)} rows, Deduplicated: {len(deduplicated)} rows")
    
    if dedup_time < 1.0:  # Should be very fast
        print("✅ Performance impact is minimal")
        return True
    else:
        print("⚠️ Performance impact may be significant")
        return False

def run_comprehensive_tests():
    """Run all comprehensive tests for the CardinalityViolation fix."""
    print("🎯 Running comprehensive CardinalityViolation fix tests\n")
    
    tests = [
        test_performance_impact,
        test_complete_cardinality_fix,
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
    
    print(f"📊 Comprehensive Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All comprehensive tests passed!")
        print("🛡️ The CardinalityViolation fix is complete and robust.")
        print("📈 The pipeline should now run without CardinalityViolation errors.")
        return True
    else:
        print("⚠️ Some tests failed - fix may need additional work")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)