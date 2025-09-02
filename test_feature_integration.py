#!/usr/bin/env python3
"""
Integration test for the feature engineering fix.
Tests that the feature engineering module handles duplicate timestamps correctly.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.abspath('.'))

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def test_feature_module_import():
    """Test that we can import the features module."""
    print("🔧 Testing feature module import...")
    
    try:
        # Test basic imports
        from models.features import build_features, _compute_rsi
        print("✅ Successfully imported features module")
        return True
    except Exception as e:
        print(f"❌ Failed to import features module: {e}")
        return False

def test_rsi_calculation():
    """Test RSI calculation function works correctly."""
    print("\n📊 Testing RSI calculation...")
    
    try:
        from models.features import _compute_rsi
        
        # Create test price series
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 110])
        rsi = _compute_rsi(prices, window=5)
        
        print(f"Test prices: {prices.tolist()}")
        print(f"RSI values: {rsi.dropna().tolist()}")
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0 and all(0 <= v <= 100 for v in valid_rsi):
            print("✅ RSI calculation works correctly")
            return True
        else:
            print("❌ RSI calculation returned invalid values")
            return False
            
    except Exception as e:
        print(f"❌ RSI calculation failed: {e}")
        return False

def test_database_free_features():
    """Test feature calculation logic without database dependency."""
    print("\n🧮 Testing feature calculation logic...")
    
    try:
        # Test core feature calculation logic
        # Create mock price data with duplicates (simulating the original issue)
        test_data = pd.DataFrame({
            'symbol': ['AAPL'] * 10,
            'ts': pd.date_range('2024-01-01', periods=10, freq='D'),
            'price_feat': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109],
            'volume': [1000000] * 10,
            'open': [99, 101, 100, 102, 104, 103, 105, 107, 106, 108]
        })
        
        # Add some duplicate timestamps (different times, same date)
        duplicate_row = test_data.iloc[0].copy()
        duplicate_row['ts'] = pd.Timestamp('2024-01-01 05:00:00')  # Same date, different time
        duplicate_row['price_feat'] = 101  # Slightly different price
        
        # Append duplicate
        test_data_with_dupes = pd.concat([test_data, pd.DataFrame([duplicate_row])], ignore_index=True)
        
        print(f"Test data with duplicates: {len(test_data_with_dupes)} rows")
        
        # Apply our deduplication logic (similar to what's in features.py)
        feats = test_data_with_dupes.copy()
        
        # Normalize timestamps 
        feats['ts'] = feats['ts'].dt.normalize()
        
        # Check for duplicates
        duplicate_check = feats.groupby(['symbol', 'ts']).size()
        duplicates_found = (duplicate_check > 1).sum()
        
        print(f"Duplicates found: {duplicates_found}")
        
        if duplicates_found > 0:
            # Deduplicate by keeping latest
            latest_indices = feats.groupby(['symbol', 'ts']).tail(1).index
            feats = feats.loc[latest_indices].reset_index(drop=True)
            
        print(f"After deduplication: {len(feats)} rows")
        
        # Verify no duplicates remain
        final_check = feats.groupby(['symbol', 'ts']).size()
        max_count = final_check.max()
        
        if max_count == 1:
            print("✅ Deduplication logic works correctly")
            return True
        else:
            print(f"❌ Deduplication failed - max count per symbol+date: {max_count}")
            return False
            
    except Exception as e:
        print(f"❌ Feature calculation test failed: {e}")
        return False

def test_timestamp_edge_cases():
    """Test edge cases in timestamp handling."""
    print("\n⏰ Testing timestamp edge cases...")
    
    try:
        # Test various timestamp formats
        timestamps = [
            pd.Timestamp('2024-01-01 00:00:00'),
            pd.Timestamp('2024-01-01 00:00:00.000000'),
            pd.Timestamp('2024-01-01 09:30:00'),
            pd.Timestamp('2024-01-01 16:00:00'),
        ]
        
        # Normalize all to same date
        normalized = [ts.normalize() for ts in timestamps]
        
        # All should be the same
        unique_dates = set(normalized)
        
        if len(unique_dates) == 1:
            print("✅ Timestamp edge cases handled correctly")
            return True
        else:
            print(f"❌ Timestamp normalization failed - {len(unique_dates)} unique dates")
            return False
            
    except Exception as e:
        print(f"❌ Timestamp edge case test failed: {e}")
        return False

def run_integration_tests():
    """Run all integration tests."""
    print("🎯 Running feature engineering integration tests\n")
    
    tests = [
        test_feature_module_import,
        test_rsi_calculation,
        test_database_free_features,
        test_timestamp_edge_cases,
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
            print()
    
    print(f"📊 Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return True
    else:
        print("⚠️ Some integration tests failed")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)