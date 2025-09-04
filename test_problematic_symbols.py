#!/usr/bin/env python3
"""Test the specific symbols mentioned in the error logs."""

import asyncio
import pandas as pd
from unittest.mock import patch
from datetime import date

# Import the function we're testing
from data.ingest import _fetch_polygon_daily_one


async def test_problematic_symbols():
    """Test the specific symbols that were causing errors."""
    print("Testing problematic symbols from error logs...")
    
    problematic_symbols = ["CCIXU", "UCB.PRI"]
    
    for symbol in problematic_symbols:
        print(f"\nTesting symbol: {symbol}")
        
        # Mock the response that would cause the original error
        with patch('data.ingest.POLYGON_API_KEY', 'test_key'), \
             patch('data.ingest.get_json_async') as mock_get:
            
            # Mock response similar to what causes the error
            mock_get.return_value = {
                "status": "OK", 
                "count": 0, 
                "resultsCount": 0,
                "adjusted": True,
                "next_url": None,
                "request_id": "test_request"
            }
            
            result = await _fetch_polygon_daily_one(symbol, date(2025, 8, 28), date(2025, 9, 4))
            
            if result.empty:
                print(f"✓ {symbol}: Scalar response handled correctly (no crash)")
            else:
                print(f"❌ {symbol}: Expected empty DataFrame, got: {result}")
                return False
    
    print("\n✓ All problematic symbols now handled safely!")
    return True


if __name__ == "__main__":
    asyncio.run(test_problematic_symbols())