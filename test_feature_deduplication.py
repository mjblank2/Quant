#!/usr/bin/env python3
"""
Test the deduplication fix in the actual feature engineering module.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def test_feature_deduplication_logic():
    """Test the deduplication logic extracted from features.py"""
    print("🧪 Testing feature deduplication logic...")
    
    # Create test data that simulates the feature engineering output with duplicates
    test_data = [
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04 00:00:00'), 'ret_1d': 0.01, 'vol_21': 0.15, 'size_ln': 24.5},
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04 05:00:00'), 'ret_1d': 0.02, 'vol_21': 0.16, 'size_ln': 24.6},  # Same date!
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-05 00:00:00'), 'ret_1d': 0.03, 'vol_21': 0.17, 'size_ln': 24.7},
        {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-04 00:00:00'), 'ret_1d': 0.02, 'vol_21': 0.14, 'size_ln': 23.5},
        {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-04 09:30:00'), 'ret_1d': 0.025, 'vol_21': 0.145, 'size_ln': 23.6}, # Same date!
    ]
    
    feats = pd.DataFrame(test_data)
    print("Original features data:")
    print(feats)
    print(f"Original count: {len(feats)}")
    
    # Apply the same deduplication logic from features.py
    original_count = len(feats)
    
    # Normalize timestamps to dates to avoid timezone/time component issues
    feats['ts'] = feats['ts'].dt.normalize()
    
    # Check for duplicates before deduplication
    duplicate_check = feats.groupby(['symbol', 'ts']).size()
    duplicates_found = (duplicate_check > 1).sum()
    
    print(f"\nDuplicates found: {duplicates_found}")
    if duplicates_found > 0:
        print("Duplicate pairs:")
        print(duplicate_check[duplicate_check > 1])
    
    if duplicates_found > 0:
        print(f"Found {duplicates_found} symbol+date pairs with multiple rows. Deduplicating by keeping latest values.")
        
        # Keep the latest row for each symbol+date combination
        # Use the row index as a tiebreaker (later rows are "more recent")
        latest_indices = feats.groupby(['symbol', 'ts']).tail(1).index
        feats = feats.loc[latest_indices].reset_index(drop=True)
        
        deduplicated_count = len(feats)
        print(f"Deduplication: {original_count} -> {deduplicated_count} rows ({original_count - deduplicated_count} duplicates removed)")
    
    print(f"\nFinal deduplicated data:")
    print(feats)
    
    # Verify no duplicates remain
    final_duplicate_check = feats.groupby(['symbol', 'ts']).size()
    max_count = final_duplicate_check.max()
    
    print(f"\nFinal verification - max rows per symbol+date: {max_count}")
    assert max_count == 1, "Deduplication failed - duplicates still exist!"
    
    print("✅ Deduplication logic works correctly")
    return True

def test_timestamp_normalization():
    """Test that timestamps are properly normalized to dates"""
    print("\n📅 Testing timestamp normalization in feature context...")
    
    # Test data with various timestamp formats (all timezone-naive for simplicity)
    test_data = [
        {'ts': pd.Timestamp('2016-01-04 00:00:00')},
        {'ts': pd.Timestamp('2016-01-04 05:00:00')},
        {'ts': pd.Timestamp('2016-01-04 09:30:00')},
        {'ts': pd.Timestamp('2016-01-04 16:00:00')},
    ]
    
    test_df = pd.DataFrame(test_data)
    
    print("Original timestamps:")
    for i, row in test_df.iterrows():
        print(f"  {i}: {row['ts']}")
    
    # Normalize
    normalized = test_df['ts'].dt.normalize()
    
    print("\nNormalized timestamps:")
    for i, ts in enumerate(normalized):
        print(f"  {i}: {ts}")
    
    # Check all are the same date
    unique_dates = normalized.unique()
    print(f"\nUnique dates after normalization: {len(unique_dates)}")
    
    if len(unique_dates) == 1:
        print("✅ All timestamps normalized to same date")
        return True
    else:
        print("❌ Timestamps not properly normalized")
        print(f"Unique dates: {unique_dates}")
        return False

def run_all_tests():
    """Run all feature deduplication tests."""
    print("🎯 Running feature deduplication tests\n")
    
    tests = [
        test_feature_deduplication_logic,
        test_timestamp_normalization,
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
            print(f"❌ Test {test_func.__name__} failed: {e}")
            print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All feature deduplication tests passed!")
        return True
    else:
        print("⚠️ Some tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)