#!/usr/bin/env python3
"""
Test script for the new ingestion functionality
"""
import os
import sys
from datetime import date, timedelta
import pandas as pd

# Set up environment
os.environ["DATABASE_URL"] = "sqlite:///test_ingestion.db"

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_max_bind_params():
    """Test the _max_bind_params_for_connection function"""
    print("Testing _max_bind_params_for_connection...")
    
    from db import engine, _max_bind_params_for_connection
    
    with engine.begin() as conn:
        max_params = _max_bind_params_for_connection(conn)
        print(f"✓ Max bind params for {conn.engine.url}: {max_params}")
        
        # For SQLite, should be 999
        if "sqlite" in str(conn.engine.url).lower():
            assert max_params == 999, f"Expected 999 for SQLite, got {max_params}"
            print("✓ SQLite parameter limit correctly detected")
        
    return True


def test_resolve_target_date():
    """Test the _resolve_target_end_date function"""
    print("Testing _resolve_target_end_date...")
    
    from data.ingest import _resolve_target_end_date
    
    # Test with no environment variable
    result = _resolve_target_end_date()
    print(f"✓ Default target date: {result}")
    
    # Test with explicit date
    explicit = date(2024, 1, 15)
    result = _resolve_target_end_date(explicit)
    assert result == explicit, f"Expected {explicit}, got {result}"
    print("✓ Explicit date handling works")
    
    # Test with environment variable
    os.environ["PIPELINE_TARGET_DATE"] = "2024-01-10"
    result = _resolve_target_end_date()
    expected = date(2024, 1, 10)
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ PIPELINE_TARGET_DATE environment variable works")
    
    # Clean up
    del os.environ["PIPELINE_TARGET_DATE"]
    
    return True


def test_universe_symbols():
    """Test the _universe_symbols function"""
    print("Testing _universe_symbols...")
    
    from data.ingest import _universe_symbols
    
    # This should return fallback symbols since we have an empty test DB
    symbols = _universe_symbols()
    print(f"✓ Universe symbols (fallback): {symbols}")
    
    # Should have the fallback symbols
    expected_fallback = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"]
    assert symbols == expected_fallback, f"Expected {expected_fallback}, got {symbols}"
    print("✓ Fallback symbols work correctly")
    
    return True


def test_standardize_bar_df():
    """Test the _standardize_bar_df function"""
    print("Testing _standardize_bar_df...")
    
    from data.ingest import _standardize_bar_df
    
    # Create test data with different provider formats
    test_data = [
        {"symbol": "AAPL", "timestamp": "2024-01-15", "o": 150.0, "h": 155.0, "l": 149.0, "c": 152.0, "v": 1000000},
        {"symbol": "MSFT", "timestamp": "2024-01-15", "o": 350.0, "h": 355.0, "l": 349.0, "c": 352.0, "v": 500000},
    ]
    
    df_in = pd.DataFrame(test_data)
    df_out = _standardize_bar_df(df_in)
    
    print(f"✓ Input columns: {list(df_in.columns)}")
    print(f"✓ Output columns: {list(df_out.columns)}")
    
    expected_cols = ["symbol", "ts", "open", "high", "low", "close", "adj_close", "volume", "vwap", "trade_count"]
    assert list(df_out.columns) == expected_cols, f"Expected {expected_cols}, got {list(df_out.columns)}"
    
    # Check data transformation
    assert len(df_out) == 2, f"Expected 2 rows, got {len(df_out)}"
    assert df_out.iloc[0]["symbol"] == "AAPL"
    assert df_out.iloc[0]["open"] == 150.0
    assert df_out.iloc[0]["adj_close"] == 152.0  # Should default to close
    
    print("✓ Standardization works correctly")
    
    return True


def test_dynamic_batching():
    """Test that upsert_dataframe uses dynamic batching"""
    print("Testing dynamic parameter batching...")
    
    from db import engine, upsert_dataframe, DailyBar, Base
    import pandas as pd
    
    # Create tables
    Base.metadata.create_all(engine)
    
    # Create test data with many columns to test parameter limits
    test_data = []
    for i in range(100):  # 100 rows
        test_data.append({
            "symbol": f"TEST{i:03d}",
            "ts": date(2024, 1, 15),
            "open": 100.0 + i,
            "high": 105.0 + i,
            "low": 95.0 + i,
            "close": 102.0 + i,
            "adj_close": 102.0 + i,
            "volume": 1000000 + i * 1000,
            "vwap": 101.0 + i,
            "trade_count": 1000 + i,
        })
    
    df = pd.DataFrame(test_data)
    
    # This should work without parameter limit errors
    try:
        upsert_dataframe(df, DailyBar, conflict_cols=["symbol", "ts"])
        print("✓ Dynamic batching handled large dataset successfully")
        
        # Verify data was inserted
        with engine.begin() as conn:
            from sqlalchemy import text
            result = conn.execute(text("SELECT COUNT(*) FROM daily_bars")).scalar()
            print(f"✓ Inserted {result} rows")
            assert result == 100, f"Expected 100 rows, got {result}"
            
    except Exception as e:
        print(f"✗ Dynamic batching failed: {e}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("Testing new ingestion and database improvements")
    print("=" * 60)
    
    tests = [
        test_max_bind_params,
        test_resolve_target_date,
        test_universe_symbols,
        test_standardize_bar_df,
        test_dynamic_batching,
    ]
    
    passed = 0
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
                print()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)