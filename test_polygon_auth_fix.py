#!/usr/bin/env python3
"""
Simple test to verify the Polygon.io authentication fix.
"""
import os
import sys
import logging

# Set up minimal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_universe_functions():
    """Test the universe functions work without actual API key."""
    try:
        # Import after setting up path
        from data.universe import _get_polygon_api_key, _add_polygon_auth, test_polygon_api_connection
        
        # Test helper functions
        print("✅ Successfully imported universe functions")
        
        # Test URL auth helper
        test_url = "https://api.polygon.io/v3/reference/tickers?market=stocks"
        result = _add_polygon_auth(test_url, "test_key")
        expected = "https://api.polygon.io/v3/reference/tickers?market=stocks&apiKey=test_key"
        assert result == expected, f"Expected {expected}, got {result}"
        print("✅ URL auth helper works correctly")
        
        # Test URL that already has params
        test_url2 = "https://api.polygon.io/v3/reference/tickers?market=stocks&limit=1000"
        result2 = _add_polygon_auth(test_url2, "test_key")
        expected2 = "https://api.polygon.io/v3/reference/tickers?market=stocks&limit=1000&apiKey=test_key"
        assert result2 == expected2, f"Expected {expected2}, got {result2}"
        print("✅ URL auth helper works with existing params")
        
        # Test URL that already has apiKey
        test_url3 = "https://api.polygon.io/v3/reference/tickers?market=stocks&apiKey=existing"
        result3 = _add_polygon_auth(test_url3, "test_key")
        assert result3 == test_url3, f"Expected unchanged URL, got {result3}"
        print("✅ URL auth helper doesn't double-add apiKey")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_key_validation():
    """Test API key validation."""
    try:
        from data.universe import _get_polygon_api_key
        
        # Save original env var
        original_key = os.environ.get("POLYGON_API_KEY")
        
        # Test missing key
        if "POLYGON_API_KEY" in os.environ:
            del os.environ["POLYGON_API_KEY"]
        
        try:
            _get_polygon_api_key()
            print("❌ Should have raised RuntimeError for missing API key")
            return False
        except RuntimeError as e:
            if "POLYGON_API_KEY environment variable is required" in str(e):
                print("✅ Correctly raises error for missing API key")
            else:
                print(f"❌ Wrong error message: {e}")
                return False
        
        # Test empty key
        os.environ["POLYGON_API_KEY"] = ""
        try:
            _get_polygon_api_key()
            print("❌ Should have raised RuntimeError for empty API key")
            return False
        except RuntimeError:
            print("✅ Correctly raises error for empty API key")
        
        # Test valid key
        os.environ["POLYGON_API_KEY"] = "test_key"
        result = _get_polygon_api_key()
        assert result == "test_key", f"Expected 'test_key', got '{result}'"
        print("✅ Correctly returns valid API key")
        
        # Restore original
        if original_key is not None:
            os.environ["POLYGON_API_KEY"] = original_key
        elif "POLYGON_API_KEY" in os.environ:
            del os.environ["POLYGON_API_KEY"]
        
        return True
        
    except Exception as e:
        print(f"❌ API key validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_connection_test():
    """Test the connection test function."""
    try:
        from data.universe import test_polygon_api_connection
        
        # Save original env var
        original_key = os.environ.get("POLYGON_API_KEY")
        
        # Test without API key
        if "POLYGON_API_KEY" in os.environ:
            del os.environ["POLYGON_API_KEY"]
        
        result = test_polygon_api_connection()
        assert result is False, "Should return False when API key is missing"
        print("✅ Connection test correctly handles missing API key")
        
        # Restore original
        if original_key is not None:
            os.environ["POLYGON_API_KEY"] = original_key
        
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Polygon.io authentication fixes...")
    
    # Test basic functions
    success = True
    success &= test_universe_functions()
    success &= test_api_key_validation()
    success &= test_connection_test()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)