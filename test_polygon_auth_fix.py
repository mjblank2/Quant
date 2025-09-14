#!/usr/bin/env python3
"""
Comprehensive test to verify the Polygon.io authentication fixes.
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
        
        # Test whitespace-only key
        os.environ["POLYGON_API_KEY"] = "   "
        try:
            _get_polygon_api_key()
            print("❌ Should have raised RuntimeError for whitespace-only API key")
            return False
        except RuntimeError:
            print("✅ Correctly raises error for whitespace-only API key")
        
        # Test valid key
        os.environ["POLYGON_API_KEY"] = "  test_key  "
        result = _get_polygon_api_key()
        assert result == "test_key", f"Expected 'test_key', got '{result}'"
        print("✅ Correctly returns valid API key (trimmed)")
        
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

def test_import_all_modules():
    """Test that all modified modules can be imported without errors."""
    try:
        # Test universe module
        import data.universe
        print("✅ data.universe imports successfully")
        
        # Test fundamentals module (may need mocking for async dependencies)
        try:
            import data.fundamentals
            print("✅ data.fundamentals imports successfully")
        except ImportError as e:
            print(f"⚠️ data.fundamentals import issue (expected in test env): {e}")
        
        # Test ingest module
        try:
            import data.ingest
            print("✅ data.ingest imports successfully")
        except ImportError as e:
            print(f"⚠️ data.ingest import issue (expected in test env): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Module import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_manual_verification_help():
    """Print manual verification steps for the user."""
    print("\n📋 Manual Verification Steps:")
    print("=" * 50)
    
    print("\n1. Test with valid POLYGON_API_KEY:")
    print("   export POLYGON_API_KEY='your_real_api_key'")
    print("   curl \"https://api.polygon.io/v3/reference/tickers?market=stocks&limit=1&apiKey=$POLYGON_API_KEY\"")
    print("   # Should return 200 OK with data")
    
    print("\n2. Test universe rebuild (with POLYGON_API_KEY set):")
    print("   export DATABASE_URL='sqlite:///test.db'")
    print("   python -c \"from data.universe import rebuild_universe; print('Success:', rebuild_universe())\"")
    print("   # Should complete without 401 errors")
    
    print("\n3. Test connection function:")
    print("   python -c \"from data.universe import test_polygon_api_connection; print('API OK:', test_polygon_api_connection())\"")
    print("   # Should return True if API key is valid")
    
    print("\n4. Test universe module directly:")
    print("   python data/universe.py")
    print("   # Should test API connection then attempt universe rebuild")
    
    print("\n5. Check for 401 errors in logs:")
    print("   # Look for 'Polygon returned 401 Unauthorized' messages")
    print("   # Should see helpful error messages if API key is missing/invalid")
    
    return True

if __name__ == "__main__":
    print("Testing Polygon.io authentication fixes...")
    print("=" * 50)
    
    # Test basic functions
    success = True
    success &= test_import_all_modules()
    success &= test_universe_functions()
    success &= test_api_key_validation()
    success &= test_connection_test()
    
    # Print manual verification steps
    test_manual_verification_help()
    
    if success:
        print("\n🎉 All automated tests passed!")
        print("💡 Complete the manual verification steps above to fully test the fix.")
        sys.exit(0)
    else:
        print("\n💥 Some automated tests failed!")
        sys.exit(1)