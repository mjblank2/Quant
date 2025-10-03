#!/usr/bin/env python3
"""
Test to verify that the chunk_size parameter fix resolves the pipeline error.

This test reproduces the exact scenario from the error:
  TypeError: upsert_dataframe() got an unexpected keyword argument 'chunk_size'
  
And verifies that the fix allows the parameter to be passed successfully.
"""
import os
import sys
import tempfile
import pandas as pd

# Set up test database
db_file = tempfile.mktemp(suffix='.db')
os.environ['DATABASE_URL'] = f'sqlite:///{db_file}'

import db


def test_chunk_size_parameter():
    """Test that upsert_dataframe accepts chunk_size parameter."""
    print("Testing chunk_size parameter fix...")
    
    # Create tables
    db.create_tables()
    
    # Create test data similar to what features.py produces
    test_data = []
    for i in range(50):
        test_data.append({
            'symbol': f'TEST{i:03d}',
            'ts': pd.Timestamp('2024-01-01'),
            'ret_1d': 0.01 * i,
            'ret_5d': 0.02 * i,
            'vol_21': 0.001 * i,
        })
    
    df = pd.DataFrame(test_data)
    
    # Test 1: The exact call from models/features.py line 362
    print("\nTest 1: Exact call from models/features.py line 362")
    print("  upsert_dataframe(feats, Feature, ['symbol', 'ts'], chunk_size=200)")
    try:
        db.upsert_dataframe(df, db.Feature, ['symbol', 'ts'], chunk_size=200)
        print("  ✅ PASS: No TypeError about chunk_size")
    except TypeError as e:
        if 'chunk_size' in str(e):
            print(f"  ❌ FAIL: Still getting chunk_size error: {e}")
            return False
        raise
    
    # Test 2: Call from features/build_features.py line 340
    df['ts'] = pd.Timestamp('2024-01-02')
    print("\nTest 2: Call from features/build_features.py line 340")
    print("  upsert_dataframe(feats, Feature, ['symbol', 'ts'], chunk_size=200)")
    try:
        db.upsert_dataframe(df, db.Feature, ['symbol', 'ts'], chunk_size=200)
        print("  ✅ PASS: No TypeError about chunk_size")
    except TypeError as e:
        if 'chunk_size' in str(e):
            print(f"  ❌ FAIL: Still getting chunk_size error: {e}")
            return False
        raise
    
    # Test 3: Call from features/store.py line 63
    df['ts'] = pd.Timestamp('2024-01-03')
    print("\nTest 3: Call from features/store.py line 63")
    print("  upsert_dataframe(features_df, Feature, ['symbol', 'ts'], chunk_size=200)")
    try:
        db.upsert_dataframe(df, db.Feature, ['symbol', 'ts'], chunk_size=200)
        print("  ✅ PASS: No TypeError about chunk_size")
    except TypeError as e:
        if 'chunk_size' in str(e):
            print(f"  ❌ FAIL: Still getting chunk_size error: {e}")
            return False
        raise
    
    # Test 4: Verify backward compatibility (without chunk_size)
    df['ts'] = pd.Timestamp('2024-01-04')
    print("\nTest 4: Backward compatibility (no chunk_size parameter)")
    print("  upsert_dataframe(df, Feature, ['symbol', 'ts'])")
    try:
        db.upsert_dataframe(df, db.Feature, ['symbol', 'ts'])
        print("  ✅ PASS: Works without chunk_size parameter")
    except Exception as e:
        print(f"  ❌ FAIL: Backward compatibility broken: {e}")
        return False
    
    # Test 5: Verify data was inserted
    print("\nTest 5: Verify data insertion")
    with db.engine.connect() as conn:
        from sqlalchemy import text
        count = conn.execute(text('SELECT COUNT(*) FROM features')).scalar()
        expected = 50 * 4  # 50 rows × 4 different dates
        if count == expected:
            print(f"  ✅ PASS: Inserted {count} rows as expected")
        else:
            print(f"  ❌ FAIL: Expected {expected} rows, got {count}")
            return False
    
    return True


def test_various_chunk_sizes():
    """Test that various chunk_size values work correctly."""
    print("\n\nTesting various chunk_size values...")
    
    test_num = 0
    for chunk_size in [50, 100, 200, 500, 1000]:
        test_num += 1
        df = pd.DataFrame({
            'symbol': [f'TEST{i:03d}' for i in range(20)],
            'ts': pd.Timestamp(f'2024-02-{test_num:02d}'),  # Different date for each test
            'ret_1d': [0.01 * i for i in range(20)],
        })
        
        try:
            db.upsert_dataframe(df, db.Feature, ['symbol', 'ts'], chunk_size=chunk_size)
            print(f"  ✅ chunk_size={chunk_size}: OK")
        except Exception as e:
            print(f"  ❌ chunk_size={chunk_size}: {e}")
            return False
    
    return True


def main():
    """Run all tests."""
    try:
        test1_passed = test_chunk_size_parameter()
        test2_passed = test_various_chunk_sizes()
        
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        if test1_passed and test2_passed:
            print("✅ ALL TESTS PASSED")
            print("\nThe chunk_size parameter fix successfully resolves the pipeline error:")
            print("  TypeError: upsert_dataframe() got an unexpected keyword argument 'chunk_size'")
            return 0
        else:
            print("❌ SOME TESTS FAILED")
            return 1
            
    finally:
        # Cleanup
        if os.path.exists(db_file):
            os.unlink(db_file)


if __name__ == "__main__":
    sys.exit(main())
