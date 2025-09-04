#!/usr/bin/env python3
"""Test the Polygon DataFrame fix in the actual ingestion code."""

import asyncio
import pandas as pd
from unittest.mock import AsyncMock, patch
from datetime import date

# Import the function we're testing
from data.ingest import _fetch_polygon_daily_one


async def test_polygon_fix_with_mock():
    """Test the fix using mocked HTTP responses."""
    print("Testing Polygon fix with mocked responses...")
    
    # Test 1: Mock response with scalar values (causes original error)
    with patch('data.ingest.POLYGON_API_KEY', 'test_key'), \
         patch('data.ingest.get_json_async') as mock_get:
        mock_get.return_value = {"status": "OK", "count": 0, "resultsCount": 0}
        
        result = await _fetch_polygon_daily_one("TESTCOIN", date(2023, 1, 1), date(2023, 1, 5))
        
        if result.empty:
            print("✓ Scalar response handled correctly (empty DataFrame)")
        else:
            print(f"❌ Expected empty DataFrame, got: {result}")
            return False
    
    # Test 2: Mock response with actual data
    with patch('data.ingest.POLYGON_API_KEY', 'test_key'), \
         patch('data.ingest.get_json_async') as mock_get:
        mock_get.return_value = {
            "status": "OK", 
            "results": [
                {"t": 1625097600000, "o": 100.0, "h": 105.0, "l": 95.0, "c": 102.0, "v": 1000}
            ]
        }
        
        result = await _fetch_polygon_daily_one("TESTSTOCK", date(2023, 1, 1), date(2023, 1, 5))
        
        if not result.empty and len(result) == 1 and 't' in result.columns:
            print(f"✓ Valid data response handled correctly: {result.shape}, columns: {list(result.columns)}")
        else:
            print(f"❌ Expected non-empty DataFrame with 't' column, got: {result}")
            print(f"   Shape: {result.shape}, Columns: {list(result.columns) if not result.empty else 'N/A'}")
            return False
    
    # Test 3: Mock response with empty results list
    with patch('data.ingest.POLYGON_API_KEY', 'test_key'), \
         patch('data.ingest.get_json_async') as mock_get:
        mock_get.return_value = {"status": "OK", "results": []}
        
        result = await _fetch_polygon_daily_one("EMPTYSYMBOL", date(2023, 1, 1), date(2023, 1, 5))
        
        if result.empty:
            print("✓ Empty results handled correctly")
        else:
            print(f"❌ Expected empty DataFrame for empty results, got: {result}")
            return False
    
    # Test 4: Mock empty response 
    with patch('data.ingest.POLYGON_API_KEY', 'test_key'), \
         patch('data.ingest.get_json_async') as mock_get:
        mock_get.return_value = {}
        
        result = await _fetch_polygon_daily_one("NULLSYMBOL", date(2023, 1, 1), date(2023, 1, 5))
        
        if result.empty:
            print("✓ Empty response handled correctly")
        else:
            print(f"❌ Expected empty DataFrame for empty response, got: {result}")
            return False
    
    # Test 5: Mock exception (should be caught and return empty DataFrame)
    with patch('data.ingest.POLYGON_API_KEY', 'test_key'), \
         patch('data.ingest.get_json_async') as mock_get:
        mock_get.side_effect = Exception("Network error")
        
        result = await _fetch_polygon_daily_one("ERRORSYMBOL", date(2023, 1, 1), date(2023, 1, 5))
        
        if result.empty:
            print("✓ Exception handled correctly")
        else:
            print(f"❌ Expected empty DataFrame for exception, got: {result}")
            return False
    
    return True


async def main():
    print("Polygon DataFrame Fix Integration Test")
    print("=" * 50)
    
    # Test the fix
    success = await test_polygon_fix_with_mock()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All integration tests passed!")
    else:
        print("❌ Some integration tests failed!")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())