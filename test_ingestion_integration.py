#!/usr/bin/env python3
"""
Integration test for the new ingestion functionality
Tests the actual ingestion process without requiring API keys
"""
import os
import sys
from datetime import date, timedelta
import pandas as pd

# Set up environment
os.environ["DATABASE_URL"] = "sqlite:///test_integration.db"

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ingestion_without_api_keys():
    """Test ingestion behavior when no API keys are available (should use fallback)"""
    print("Testing ingestion without API keys...")
    
    # Make sure no API keys are set
    for key in ["APCA_API_KEY_ID", "ALPACA_API_KEY", "POLYGON_API_KEY"]:
        if key in os.environ:
            del os.environ[key]
    
    from data.ingest import ingest_bars_for_universe
    from db import Base, engine
    
    # Create tables
    Base.metadata.create_all(engine)
    
    # This should fail gracefully since we have no API keys
    try:
        result = ingest_bars_for_universe(days=1)
        print(f"✓ Ingestion completed: {result}")
        # It should return False since no data providers are available
        assert result is False, "Expected False when no API keys are available"
        print("✓ Correctly handled missing API keys")
    except Exception as e:
        print(f"✗ Ingestion failed unexpectedly: {e}")
        return False
    
    return True


def test_standardization_integration():
    """Test the standardization functions work with real-world-like data"""
    print("Testing standardization with realistic data...")
    
    from data.ingest import _standardize_bar_df
    
    # Test with Alpaca-style data
    alpaca_data = pd.DataFrame([
        {
            "symbol": "AAPL",
            "timestamp": "2024-01-15T00:00:00+00:00",
            "open": 150.0,
            "high": 155.0,
            "low": 149.0,
            "close": 152.0,
            "volume": 1000000,
            "vwap": 151.5,
            "trade_count": 5000,
        }
    ])
    
    standardized = _standardize_bar_df(alpaca_data)
    print(f"✓ Alpaca-style standardization: {len(standardized)} rows")
    assert len(standardized) == 1
    assert standardized.iloc[0]["symbol"] == "AAPL"
    assert standardized.iloc[0]["open"] == 150.0
    
    # Test with Polygon-style data (using different column names)
    polygon_data = pd.DataFrame([
        {
            "symbol": "MSFT",
            "t": "2024-01-15",
            "o": 350.0,
            "h": 355.0,
            "l": 349.0,
            "c": 352.0,
            "v": 500000,
            "vw": 351.5,
            "n": 3000,
        }
    ])
    
    standardized = _standardize_bar_df(polygon_data)
    print(f"✓ Polygon-style standardization: {len(standardized)} rows")
    print(f"  Debug: trade_count = {standardized.iloc[0]['trade_count']}")
    assert len(standardized) == 1
    assert standardized.iloc[0]["symbol"] == "MSFT"
    assert standardized.iloc[0]["open"] == 350.0
    assert standardized.iloc[0]["vwap"] == 351.5
    # The column mapping should convert "n" to "trade_count"
    assert standardized.iloc[0]["trade_count"] == 3000, f"Expected 3000, got {standardized.iloc[0]['trade_count']}"
    
    print("✓ Multi-provider standardization works correctly")
    return True


def test_date_resolution():
    """Test the date resolution functionality"""
    print("Testing date resolution...")
    
    from data.ingest import _resolve_target_end_date
    
    # Test normal case
    result = _resolve_target_end_date()
    print(f"✓ Normal date resolution: {result}")
    
    # Test with environment variable
    os.environ["PIPELINE_TARGET_DATE"] = "2024-01-15"
    result = _resolve_target_end_date()
    expected = date(2024, 1, 15)
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Environment variable date resolution works")
    
    # Clean up
    del os.environ["PIPELINE_TARGET_DATE"]
    
    return True


def test_database_integration():
    """Test that the database integration works correctly"""
    print("Testing database integration...")
    
    from db import Base, engine, upsert_dataframe, DailyBar
    from data.ingest import _standardize_bar_df
    
    # Create tables
    Base.metadata.create_all(engine)
    
    # Create test data
    test_data = pd.DataFrame([
        {
            "symbol": "TEST",
            "timestamp": "2024-01-15",
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000000,
        }
    ])
    
    # Standardize
    standardized = _standardize_bar_df(test_data)
    
    # Upsert
    upsert_dataframe(standardized, DailyBar, conflict_cols=["symbol", "ts"])
    
    # Verify
    from sqlalchemy import text
    with engine.begin() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM daily_bars WHERE symbol = 'TEST'")).scalar()
        assert result == 1, f"Expected 1 row, got {result}"
        
        # Test upsert (should update, not insert)
        upsert_dataframe(standardized, DailyBar, conflict_cols=["symbol", "ts"])
        result = conn.execute(text("SELECT COUNT(*) FROM daily_bars WHERE symbol = 'TEST'")).scalar()
        assert result == 1, f"Expected 1 row after upsert, got {result}"
    
    print("✓ Database integration works correctly")
    return True


def main():
    """Run all integration tests"""
    print("Running ingestion integration tests")
    print("=" * 50)
    
    tests = [
        test_date_resolution,
        test_standardization_integration,
        test_database_integration,
        test_ingestion_without_api_keys,
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
    
    print("=" * 50)
    print(f"Integration tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("✓ All integration tests passed!")
        return True
    else:
        print("✗ Some integration tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)