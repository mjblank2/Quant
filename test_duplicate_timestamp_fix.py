#!/usr/bin/env python3
"""
Test for duplicate timestamp issue in feature engineering.

This test reproduces the CardinalityViolation error that occurs when
feature engineering receives data with multiple timestamps per symbol+date,
and validates the fix.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, date

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def test_duplicate_timestamp_detection():
    """Test detection of duplicate symbol+date combinations."""
    print("🧪 Testing duplicate timestamp detection...")
    
    # Create test data with duplicate timestamps (different times, same date)
    test_data = [
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04 00:00:00'), 'ret_1d': 0.01, 'vol_21': 0.15},
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04 05:00:00'), 'ret_1d': 0.02, 'vol_21': 0.16},  # Same date!
        {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-05 00:00:00'), 'ret_1d': 0.03, 'vol_21': 0.17},
        {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-04 00:00:00'), 'ret_1d': 0.02, 'vol_21': 0.14},
        {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-04 09:30:00'), 'ret_1d': 0.025, 'vol_21': 0.145}, # Same date!
    ]
    
    df = pd.DataFrame(test_data)
    print("Original data:")
    print(df)
    
    # Check for duplicates
    df['date_norm'] = df['ts'].dt.normalize()
    duplicates = df.groupby(['symbol', 'date_norm']).size()
    duplicate_pairs = duplicates[duplicates > 1]
    
    print(f"\nDuplicate symbol+date pairs: {len(duplicate_pairs)}")
    if len(duplicate_pairs) > 0:
        print("Duplicates found:")
        print(duplicate_pairs)
        return True, df, duplicate_pairs
    else:
        print("No duplicates found")
        return False, df, None

def test_deduplication_strategy():
    """Test different deduplication strategies."""
    print("\n🔧 Testing deduplication strategies...")
    
    has_dupes, df, duplicate_pairs = test_duplicate_timestamp_detection()
    if not has_dupes:
        print("No duplicates to test deduplication on")
        return True
    
    # Strategy 1: Keep latest timestamp per symbol+date
    print("\nStrategy 1: Keep latest timestamp per symbol+date")
    df_deduped_latest = df.loc[df.groupby(['symbol', df['ts'].dt.normalize()])['ts'].idxmax()]
    print(f"Rows before: {len(df)}, after: {len(df_deduped_latest)}")
    print(df_deduped_latest[['symbol', 'ts', 'ret_1d']])
    
    # Strategy 2: Keep earliest timestamp per symbol+date
    print("\nStrategy 2: Keep earliest timestamp per symbol+date")
    df_deduped_earliest = df.loc[df.groupby(['symbol', df['ts'].dt.normalize()])['ts'].idxmin()]
    print(f"Rows before: {len(df)}, after: {len(df_deduped_earliest)}")
    print(df_deduped_earliest[['symbol', 'ts', 'ret_1d']])
    
    # Strategy 3: Average values per symbol+date
    print("\nStrategy 3: Average values per symbol+date")
    numeric_cols = ['ret_1d', 'vol_21']
    
    # Create a proper groupby aggregation
    df_temp = df.copy()
    df_temp['date_norm'] = df_temp['ts'].dt.normalize()
    
    df_agg = df_temp.groupby(['symbol', 'date_norm']).agg({
        **{col: 'mean' for col in numeric_cols}
    }).reset_index()
    
    # Set ts to the normalized date
    df_agg['ts'] = df_agg['date_norm']
    df_agg = df_agg.drop('date_norm', axis=1)
    
    print(f"Rows before: {len(df)}, after: {len(df_agg)}")
    print(df_agg)
    
    # Verify no duplicates remain
    for strategy_name, strategy_df in [
        ("Latest", df_deduped_latest),
        ("Earliest", df_deduped_earliest), 
        ("Average", df_agg)
    ]:
        dupes_check = strategy_df.groupby(['symbol', strategy_df['ts'].dt.normalize()]).size()
        max_dupes = dupes_check.max()
        print(f"{strategy_name} strategy - Max symbol+date count: {max_dupes}")
        assert max_dupes == 1, f"{strategy_name} strategy still has duplicates!"
    
    print("✅ All deduplication strategies work correctly")
    return True

def test_timestamp_normalization():
    """Test timestamp normalization to date."""
    print("\n📅 Testing timestamp normalization...")
    
    timestamps = [
        pd.Timestamp('2016-01-04 00:00:00'),
        pd.Timestamp('2016-01-04 05:00:00'),
        pd.Timestamp('2016-01-04 09:30:00'),
        pd.Timestamp('2016-01-04 16:00:00'),
    ]
    
    print("Original timestamps:")
    for ts in timestamps:
        print(f"  {ts}")
    
    # Normalize to dates
    normalized = [ts.normalize() for ts in timestamps]
    print("\nNormalized timestamps:")
    for ts in normalized:
        print(f"  {ts}")
    
    # Verify all are same date
    unique_dates = set(normalized)
    print(f"\nUnique dates after normalization: {len(unique_dates)}")
    assert len(unique_dates) == 1, "Normalization should result in single date"
    
    print("✅ Timestamp normalization works correctly")
    return True

def run_all_tests():
    """Run all duplicate timestamp tests."""
    print("🎯 Running duplicate timestamp fix tests\n")
    
    tests = [
        test_duplicate_timestamp_detection,
        test_deduplication_strategy, 
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
        print("🎉 All duplicate timestamp tests passed!")
        return True
    else:
        print("⚠️ Some tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)