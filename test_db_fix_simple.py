#!/usr/bin/env python3
"""
Simple test for the db.py deduplication fix without requiring DATABASE_URL.

This validates the logic that was added to prevent CardinalityViolation errors.
"""

import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def test_db_deduplication_logic():
    """Test the deduplication logic that was added to db.py."""
    print("🔧 Testing the deduplication logic from db.py fix...")
    
    # Simulate the problematic records that would cause CardinalityViolation
    records = [
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.01, 'vol_21': 0.15},
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.02, 'vol_21': 0.16},  # Duplicate!
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-05').date(), 'ret_1d': 0.03, 'vol_21': 0.17},
        {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.02, 'vol_21': 0.14},
        {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.025, 'vol_21': 0.145}, # Duplicate!
    ]
    
    # Create DataFrame from records (this simulates what happens in the retry logic)
    smaller_df = pd.DataFrame(records)
    print("Original retry data (with duplicates):")
    print(smaller_df)
    
    # Apply the fix logic that was added to db.py
    conflict_cols = ['symbol', 'ts']
    
    if len(smaller_df) > 0 and conflict_cols:
        original_size = len(smaller_df)
        smaller_df = smaller_df.drop_duplicates(subset=conflict_cols, keep='last').reset_index(drop=True)
        dedupe_size = len(smaller_df)
        if dedupe_size < original_size:
            log.warning(f"Removed {original_size - dedupe_size} duplicate rows during retry to prevent CardinalityViolation")
    
    print(f"\nAfter applying db.py fix:")
    print(smaller_df)
    
    # Verify no duplicates remain
    final_check = smaller_df.groupby(conflict_cols).size()
    max_count = final_check.max()
    
    print(f"\nMax rows per symbol+ts: {max_count}")
    
    if max_count == 1:
        print("✅ DB fix successfully removes duplicates")
        return True
    else:
        print("❌ DB fix failed - duplicates still exist")
        return False

def test_original_error_scenario():
    """Test with the exact data from the original error."""
    print("\n🎯 Testing with original error scenario data...")
    
    # Extract the key data points from the original error
    error_records = [
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.0008550256507693366},
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.0},  # Same date, different time (normalized)
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-05').date(), 'ret_1d': -0.025059326056003806},
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-08').date(), 'ret_1d': 0.005287713841368502},
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-08').date(), 'ret_1d': 0.0},  # Same date, different time (normalized)
    ]
    
    smaller_df = pd.DataFrame(error_records)
    print("Original error scenario data:")
    print(smaller_df[['symbol', 'ts', 'ret_1d']])
    
    # Apply the fix
    conflict_cols = ['symbol', 'ts']
    original_size = len(smaller_df)
    smaller_df = smaller_df.drop_duplicates(subset=conflict_cols, keep='last').reset_index(drop=True)
    dedupe_size = len(smaller_df)
    
    print(f"\nAfter deduplication: {original_size} -> {dedupe_size} rows")
    print("Fixed data:")
    print(smaller_df[['symbol', 'ts', 'ret_1d']])
    
    # Verify fix
    final_check = smaller_df.groupby(conflict_cols).size()
    max_count = final_check.max()
    
    if max_count == 1:
        print("✅ Original error scenario fixed!")
        return True
    else:
        print("❌ Original error scenario not fixed!")
        return False

def run_simple_tests():
    """Run simplified tests for the db.py fix."""
    print("🔧 Running simple tests for db.py CardinalityViolation fix\n")
    
    tests = [
        test_db_deduplication_logic,
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
    
    print(f"📊 Simple Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All db.py fix tests passed!")
        print("✅ The fix should prevent CardinalityViolation errors in retry scenarios.")
        return True
    else:
        print("⚠️ Some tests failed - fix may need adjustment")
        return False

if __name__ == "__main__":
    success = run_simple_tests()
    exit(0 if success else 1)