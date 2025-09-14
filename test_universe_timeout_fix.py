#!/usr/bin/env python3
"""
Test for the universe timeout fix - verifies that robust HTTP handling is working
"""
import os
import unittest
from unittest.mock import patch
import logging

# Set up test environment
os.environ["DATABASE_URL"] = "sqlite:///test.db"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TestUniverseTimeoutFix(unittest.TestCase):

    def test_robust_http_import(self):
        """Test that robust HTTP utility is imported correctly"""
        from data.universe import HAS_UTILS_HTTP, _robust_get_json
        self.assertTrue(HAS_UTILS_HTTP, "utils_http should be available")

        # Test the function exists and is callable
        self.assertTrue(callable(_robust_get_json))

    def test_robust_http_fallback(self):
        """Test that robust HTTP handles failures gracefully"""
        from data.universe import _robust_get_json

        # Test with a non-existent URL - should return empty dict without throwing
        result = _robust_get_json("https://nonexistent.example.com/test", timeout=1.0)
        self.assertEqual(result, {})

    def test_polygon_functions_exist(self):
        """Test that all the modified functions exist and are callable"""
        from data.universe import (
            _get_polygon_api_key,
            _add_polygon_auth,
            _robust_get_json,
            _list_small_cap_symbols,
            test_polygon_api_connection,
            rebuild_universe
        )

        functions = [
            _get_polygon_api_key,
            _add_polygon_auth,
            _robust_get_json,
            _list_small_cap_symbols,
            test_polygon_api_connection,
            rebuild_universe
        ]

        for func in functions:
            self.assertTrue(callable(func), f"{func.__name__} should be callable")

    @patch('data.universe._robust_get_json')
    def test_list_small_cap_with_mock(self, mock_get_json):
        """Test _list_small_cap_symbols with mocked HTTP calls"""
        from data.universe import _list_small_cap_symbols

        # Mock the API responses
        # First call (main tickers list)
        mock_get_json.side_effect = [
            {
                "results": [
                    {"ticker": "TEST1", "name": "Test Corp 1"},
                    {"ticker": "TEST2", "name": "Test Corp 2"}
                ],
                "next_url": None
            },
            # Detail calls for each ticker
            {
                "results": {"market_cap": 1_000_000_000}  # 1B - under 3B threshold
            },
            {
                "results": {"market_cap": 5_000_000_000}  # 5B - over 3B threshold
            }
        ]

        # Set a mock API key
        with patch.dict(os.environ, {"POLYGON_API_KEY": "test_key"}):
            result = _list_small_cap_symbols()

        # Should only return TEST1 (under threshold)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbol"], "TEST1")
        self.assertEqual(result[0]["name"], "Test Corp 1")

        # Verify robust HTTP was called appropriately
        self.assertEqual(mock_get_json.call_count, 3)  # 1 main + 2 detail calls

    def test_api_key_validation(self):
        """Test API key validation"""
        from data.universe import _get_polygon_api_key

        # Test missing key
        with patch.dict(os.environ, {}, clear=True):
            if "POLYGON_API_KEY" in os.environ:
                del os.environ["POLYGON_API_KEY"]
            with self.assertRaises(RuntimeError) as cm:
                _get_polygon_api_key()
            self.assertIn("POLYGON_API_KEY environment variable is required", str(cm.exception))

        # Test empty key
        with patch.dict(os.environ, {"POLYGON_API_KEY": ""}):
            with self.assertRaises(RuntimeError) as cm:
                _get_polygon_api_key()
            self.assertIn("POLYGON_API_KEY environment variable is required", str(cm.exception))

        # Test valid key
        with patch.dict(os.environ, {"POLYGON_API_KEY": "  test_key  "}):
            result = _get_polygon_api_key()
            self.assertEqual(result, "test_key")

    @patch('data.universe._robust_get_json')
    def test_connection_test_with_mock(self, mock_get_json):
        """Test API connection test with mocked response"""
        from data.universe import test_polygon_api_connection

        # Mock successful response
        mock_get_json.return_value = {
            "results": [{"ticker": "AAPL", "name": "Apple Inc."}]
        }

        with patch.dict(os.environ, {"POLYGON_API_KEY": "test_key"}):
            result = test_polygon_api_connection()

        self.assertTrue(result)
        mock_get_json.assert_called_once()

        # Test failure case
        mock_get_json.return_value = {}

        with patch.dict(os.environ, {"POLYGON_API_KEY": "test_key"}):
            result = test_polygon_api_connection()

        self.assertFalse(result)


if __name__ == "__main__":
    print("Testing Universe Timeout Fix...")
    print("=" * 50)

    # Run the tests
    unittest.main(verbosity=2)
