#!/usr/bin/env python3
"""Debug the Polygon DataFrame creation."""

import pandas as pd


def debug_polygon_responses():
    """Debug different types of Polygon responses."""
    print("Debugging Polygon response handling...")
    
    # Test different response formats
    responses = [
        # Scalar response (causes original error)
        {"status": "OK", "count": 0, "resultsCount": 0},
        
        # Valid data response
        {
            "status": "OK", 
            "results": [
                {"t": 1625097600000, "o": 100.0, "h": 105.0, "l": 95.0, "c": 102.0, "v": 1000}
            ]
        },
        
        # Empty results
        {"status": "OK", "results": []},
        
        # None response
        None,
        
        # Empty dict
        {},
    ]
    
    for i, js in enumerate(responses):
        print(f"\nResponse {i+1}: {js}")
        
        # Test original approach (would fail on first one)
        if js and not (isinstance(js, dict) and all(isinstance(v, (str, int, float, bool)) for v in js.values())):
            try:
                original_df = pd.DataFrame(js or {})
                print(f"  Original approach: {original_df.shape}")
            except Exception as e:
                print(f"  Original approach error: {e}")
        else:
            print("  Original approach: would fail with scalar values")
        
        # Test new approach
        if not js:
            df = pd.DataFrame()
            print(f"  New approach: {df.shape} (empty response)")
        elif 'results' in js and isinstance(js['results'], list) and len(js['results']) > 0:
            df = pd.DataFrame(js['results'])
            print(f"  New approach: {df.shape} (from results)")
            print(f"  Columns: {list(df.columns)}")
        else:
            df = pd.DataFrame()
            print(f"  New approach: {df.shape} (no valid results)")


if __name__ == "__main__":
    debug_polygon_responses()