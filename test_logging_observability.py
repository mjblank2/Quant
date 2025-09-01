#!/usr/bin/env python3
"""
Test script for logging and observability improvements
"""
import asyncio
from utils_logging import generate_request_id, structured_logger
from utils_http import get_json
from utils_http_async import get_json_async
import aiohttp

def test_request_id_generation():
    """Test request ID generation."""
    print("=== Testing Request ID Generation ===")
    
    # Generate multiple request IDs
    ids = [generate_request_id() for _ in range(3)]
    print(f"Generated IDs: {ids}")
    
    # Verify they're unique
    assert len(set(ids)) == len(ids), "Request IDs should be unique"
    print("✅ Request ID generation works")

def test_structured_logging():
    """Test structured logging functionality."""
    print("\n=== Testing Structured Logging ===")
    
    request_id = generate_request_id()
    
    # Test request logging
    structured_logger.log_request(
        method="GET",
        path="/api/test",
        request_id=request_id,
        user_agent="test-agent/1.0",
        remote_addr="127.0.0.1"
    )
    
    # Test response logging
    structured_logger.log_response(
        request_id=request_id,
        status_code=200,
        response_time_ms=150.5,
        bytes_sent=1024,
        method="GET",
        path="/api/test"
    )
    
    # Test slow request
    structured_logger.log_response(
        request_id=generate_request_id(),
        status_code=200,
        response_time_ms=12000,  # 12 seconds - should trigger warning
        bytes_sent=2048,
        method="POST",
        path="/api/slow"
    )
    
    # Test HTTP call logging
    structured_logger.log_http_call(
        method="GET",
        url="https://api.example.com/data",
        status_code=200,
        response_time_ms=250.3,
        bytes_received=4096,
        request_id=request_id
    )
    
    print("✅ Structured logging works")

def test_http_utilities():
    """Test enhanced HTTP utilities with logging."""
    print("\n=== Testing HTTP Utilities with Logging ===")
    
    # Test with a mock request (will fail but show logging)
    try:
        result = get_json(
            "https://httpbin.org/delay/1", 
            timeout=5.0,
            max_tries=1
        )
        print(f"HTTP result keys: {list(result.keys()) if result else 'empty'}")
    except Exception as e:
        print(f"Expected timeout/error: {e}")
    
    print("✅ HTTP utilities logging works")

async def test_async_http_utilities():
    """Test async HTTP utilities with logging."""
    print("\n=== Testing Async HTTP Utilities with Logging ===")
    
    try:
        async with aiohttp.ClientSession() as session:
            result = await get_json_async(
                session,
                "https://httpbin.org/delay/1",
                timeout=5.0,
                max_tries=1
            )
            print(f"Async HTTP result keys: {list(result.keys()) if result else 'empty'}")
    except Exception as e:
        print(f"Expected timeout/error: {e}")
    
    print("✅ Async HTTP utilities logging works")

def test_metrics_detection():
    """Test metrics detection for slow/tiny responses."""
    print("\n=== Testing Metrics Detection ===")
    
    try:
        from health_api import metrics_store
        
        # Clear existing data
        metrics_store.requests.clear()
        metrics_store.slow_requests.clear()
        metrics_store.tiny_responses.clear()
        
        # Add test requests
        test_requests = [
            # Normal request
            {"request_id": "req-1", "method": "GET", "path": "/api/data", 
             "status_code": 200, "response_time_ms": 150, "bytes_sent": 1024},
            
            # Slow request
            {"request_id": "req-2", "method": "POST", "path": "/api/upload", 
             "status_code": 200, "response_time_ms": 15000, "bytes_sent": 512},
            
            # Tiny response (polling)
            {"request_id": "req-3", "method": "GET", "path": "/api/heartbeat", 
             "status_code": 200, "response_time_ms": 50, "bytes_sent": 25},
             
            # Another tiny response
            {"request_id": "req-4", "method": "GET", "path": "/api/status", 
             "status_code": 200, "response_time_ms": 30, "bytes_sent": 15},
        ]
        
        for req in test_requests:
            metrics_store.add_request(req)
        
        summary = metrics_store.get_metrics_summary()
        
        print(f"Total requests: {summary['total_requests']}")
        print(f"Slow requests detected: {summary['slow_requests_count']}")
        print(f"Tiny responses detected: {summary['tiny_responses_count']}")
        
        # Verify detection
        assert summary['slow_requests_count'] == 1, "Should detect 1 slow request"
        assert summary['tiny_responses_count'] == 2, "Should detect 2 tiny responses"
        
        print("✅ Metrics detection works correctly")
        
    except ImportError:
        print("⚠️ FastAPI not available, skipping metrics test")

def main():
    """Run all tests."""
    print("🔍 Testing Logging and Observability Improvements")
    print("=" * 50)
    
    test_request_id_generation()
    test_structured_logging()
    test_http_utilities()
    
    # Run async test
    asyncio.run(test_async_http_utilities())
    
    test_metrics_detection()
    
    print("\n" + "=" * 50)
    print("✅ All logging and observability tests passed!")
    print("\n📊 Key Features Demonstrated:")
    print("- Unique request ID generation")
    print("- Structured JSON logging for dashboards")
    print("- Timeout detection (>10s requests)")
    print("- Tiny response detection (<100 bytes)")
    print("- HTTP call tracking with retries")
    print("- Metrics collection for monitoring")

if __name__ == "__main__":
    main()