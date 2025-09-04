#!/usr/bin/env python3
"""Test script to reproduce and fix the Polygon DataFrame error."""

import pandas as pd


def test_polygon_scalar_error():
    """Reproduce the error with scalar values."""
    print("Testing Polygon scalar value error...")
    
    # This reproduces the error from the logs
    js_with_scalars = {"status": "OK", "count": 0, "resultsCount": 0}
    
    try:
        df = pd.DataFrame(js_with_scalars)
        print("❌ Expected error but got DataFrame:", df)
        return False
    except ValueError as e:
        if "If using all scalar values, you must pass an index" in str(e):
            print("✓ Successfully reproduced the error:", str(e))
            return True
        else:
            print("❌ Got different error:", str(e))
            return False


def test_polygon_list_works():
    """Test that list data works fine."""
    print("\nTesting with list data (should work)...")
    
    js_with_list = {
        "results": [
            {"t": 1625097600000, "o": 100, "h": 105, "l": 95, "c": 102, "v": 1000}
        ]
    }
    
    try:
        df = pd.DataFrame(js_with_list)
        print("✓ List data works:", df.shape)
        return True
    except Exception as e:
        print("❌ List data failed:", str(e))
        return False


def test_fixed_approach():
    """Test the proposed fix."""
    print("\nTesting fixed approach...")
    
    def safe_polygon_dataframe(js):
        """Safe version of creating DataFrame from Polygon response."""
        if not js:
            return pd.DataFrame()
        
        # Check if there's a 'results' field with list data
        if 'results' in js and isinstance(js['results'], list) and len(js['results']) > 0:
            return pd.DataFrame(js['results'])
        
        # If no results or empty results, return empty DataFrame
        return pd.DataFrame()
    
    # Test with scalar data (should return empty DataFrame)
    js_scalars = {"status": "OK", "count": 0, "resultsCount": 0}
    df1 = safe_polygon_dataframe(js_scalars)
    print(f"✓ Scalar data handled safely: {df1.shape}")
    
    # Test with list data (should return DataFrame with data)
    js_list = {
        "results": [
            {"t": 1625097600000, "o": 100, "h": 105, "l": 95, "c": 102, "v": 1000}
        ]
    }
    df2 = safe_polygon_dataframe(js_list)
    print(f"✓ List data handled correctly: {df2.shape}")
    
    # Test with empty results
    js_empty = {"results": []}
    df3 = safe_polygon_dataframe(js_empty)
    print(f"✓ Empty results handled: {df3.shape}")
    
    return df1.empty and not df2.empty and df3.empty


if __name__ == "__main__":
    print("Polygon DataFrame Fix Test")
    print("=" * 40)
    
    success = True
    success &= test_polygon_scalar_error()
    success &= test_polygon_list_works()
    success &= test_fixed_approach()
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed!")
    else:
        print("❌ Some tests failed!")