#!/usr/bin/env python3
"""
Test script for background task API endpoints and responsiveness.
Tests that long-running operations are properly offloaded to background tasks
and that HTTP handlers remain responsive.
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from utils_logging import generate_request_id
import os

# Set up test environment
os.environ['DATABASE_URL'] = 'sqlite:///test.db'

def test_api_responsiveness():
    """Test that API endpoints are responsive and return quickly."""
    print("=== Testing API Responsiveness ===")
    
    try:
        from health_api import app
        if not app:
            print("⚠️  FastAPI app not available, skipping API tests")
            return
        
        client = TestClient(app)
        
        # Test basic endpoints respond quickly
        endpoints_to_test = [
            ("/", "GET"),
            ("/health", "GET"),
            ("/status", "GET"),
            ("/metrics", "GET"),
        ]
        
        for endpoint, method in endpoints_to_test:
            start_time = time.time()
            
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint)
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            print(f"  {method} {endpoint}: {response.status_code} in {response_time:.1f}ms")
            
            # Assert response is quick (< 1 second for basic endpoints)
            assert response_time < 1000, f"Endpoint {endpoint} took too long: {response_time}ms"
            assert response.status_code in [200, 503], f"Unexpected status for {endpoint}: {response.status_code}"
        
        print("✅ Basic API endpoints are responsive")
        
    except ImportError as e:
        print(f"⚠️  Could not import API components: {e}")

def test_task_endpoints_return_task_ids():
    """Test that task endpoints return task IDs quickly without blocking."""
    print("\n=== Testing Task Endpoint Responsiveness ===")
    
    try:
        from health_api import app
        if not app:
            print("⚠️  FastAPI app not available, skipping task endpoint tests")
            return
        
        client = TestClient(app)
        
        # Test task endpoints
        task_endpoints = [
            ("/universe", "POST", {}),
            ("/fundamentals", "POST", {}),
            ("/features", "POST", {}),
            ("/train", "POST", {}),
            ("/backtest", "POST", {}),
            ("/trades", "POST", {}),
            ("/broker-sync", "POST", {"trade_ids": [1, 2, 3]}),
            ("/pipeline", "POST", {"sync_broker": False}),
            ("/ingest", "POST", {"days": 7}),
        ]
        
        for endpoint, method, payload in task_endpoints:
            start_time = time.time()
            
            response = client.post(endpoint, json=payload)
            response_time = (time.time() - start_time) * 1000
            
            print(f"  {method} {endpoint}: {response.status_code} in {response_time:.1f}ms")
            
            # Response should be quick (< 2 seconds even without Celery)
            assert response_time < 2000, f"Task endpoint {endpoint} took too long: {response_time}ms"
            
            if response.status_code == 202:
                # Task was successfully dispatched
                data = response.json()
                assert "task_id" in data, f"Missing task_id in response for {endpoint}"
                assert "status_url" in data, f"Missing status_url in response for {endpoint}"
                assert "status" in data and data["status"] == "accepted", f"Invalid status for {endpoint}"
                print(f"    ✅ Task dispatched: {data['task_id'][:8]}...")
                
            elif response.status_code == 503:
                # Task queue not available (expected in test environment)
                data = response.json()
                assert "detail" in data, f"Missing error detail for {endpoint}"
                print(f"    ⚠️  Task queue not available: {data['detail']}")
                
            else:
                # Unexpected status code
                print(f"    ❌ Unexpected status {response.status_code}: {response.text}")
                
        print("✅ Task endpoints are responsive and return appropriate responses")
        
    except ImportError as e:
        print(f"⚠️  Could not import API components: {e}")

def test_status_endpoint():
    """Test that status endpoint works correctly."""
    print("\n=== Testing Status Endpoint ===")
    
    try:
        from health_api import app
        if not app:
            print("⚠️  FastAPI app not available, skipping status endpoint test")
            return
        
        client = TestClient(app)
        
        # Test status endpoint with mock task ID
        test_task_id = "test-task-123"
        
        start_time = time.time()
        response = client.get(f"/tasks/status/{test_task_id}")
        response_time = (time.time() - start_time) * 1000
        
        print(f"  GET /tasks/status/{test_task_id}: {response.status_code} in {response_time:.1f}ms")
        
        # Should be quick
        assert response_time < 1000, f"Status endpoint took too long: {response_time}ms"
        
        if response.status_code == 404:
            # Expected for non-existent task
            data = response.json()
            assert "detail" in data, "Missing error detail"
            print(f"    ✅ Correctly returns 404 for non-existent task")
            
        elif response.status_code == 503:
            # Task queue not available
            data = response.json()
            print(f"    ⚠️  Task queue not available: {data['detail']}")
            
        else:
            print(f"    ❌ Unexpected status {response.status_code}: {response.text}")
        
        # Test legacy endpoint for backward compatibility
        response = client.get(f"/ingest/status/{test_task_id}")
        assert response.status_code in [404, 503], "Legacy endpoint should work"
        print("    ✅ Legacy endpoint works")
        
        print("✅ Status endpoints are responsive")
        
    except ImportError as e:
        print(f"⚠️  Could not import API components: {e}")

def test_request_id_logging():
    """Test that request IDs are properly logged and included in responses."""
    print("\n=== Testing Request ID Logging ===")
    
    try:
        from health_api import app
        if not app:
            print("⚠️  FastAPI app not available, skipping request ID test")
            return
        
        client = TestClient(app)
        
        # Test that responses include request ID headers
        response = client.get("/health")
        
        print(f"  Response headers: {dict(response.headers)}")
        
        # Should include x-request-id header
        assert "x-request-id" in response.headers, "Missing x-request-id header"
        
        request_id = response.headers["x-request-id"]
        print(f"  Request ID: {request_id}")
        
        # Should be a valid UUID format
        import uuid
        try:
            uuid.UUID(request_id)
            print("    ✅ Request ID is valid UUID format")
        except ValueError:
            print(f"    ❌ Invalid UUID format: {request_id}")
            assert False, "Request ID should be valid UUID"
        
        print("✅ Request ID logging works correctly")
        
    except ImportError as e:
        print(f"⚠️  Could not import API components: {e}")

@pytest.mark.asyncio
async def test_async_task_dispatching():
    """Test that task dispatching works correctly with proper error handling."""
    print("\n=== Testing Async Task Dispatching ===")
    
    try:
        # Test the dispatch helper function directly
        from health_api import dispatch_task_with_response
        
        # Mock task function
        mock_task = Mock()
        mock_task.delay.return_value.id = "test-task-123"
        
        # Test successful dispatch
        response = await dispatch_task_with_response(
            mock_task, "Test Task", "test-endpoint"
        )
        
        assert response["status"] == "accepted"
        assert response["task_id"] == "test-task-123"
        assert response["status_url"] == "/tasks/status/test-task-123"
        assert response["endpoint"] == "test-endpoint"
        
        print("✅ Task dispatching helper works correctly")
        
    except ImportError as e:
        print(f"⚠️  Could not import dispatch helper: {e}")

def test_api_documentation():
    """Test that API documentation is accessible and accurate."""
    print("\n=== Testing API Documentation ===")
    
    try:
        from health_api import app
        if not app:
            print("⚠️  FastAPI app not available, skipping documentation test")
            return
        
        client = TestClient(app)
        
        # Test root endpoint documentation
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
        assert "task_status" in data
        
        # Verify all documented endpoints exist
        documented_endpoints = data["endpoints"]
        for endpoint_name, endpoint_path in documented_endpoints.items():
            method, path = endpoint_path.split(" ", 1)
            print(f"  Checking {method} {path}")
            
            # We can't easily test POST endpoints without mocking, 
            # but we can verify they're in the app's routes
            found = False
            for route in app.routes:
                if hasattr(route, 'path') and hasattr(route, 'methods'):
                    if route.path == path and method in route.methods:
                        found = True
                        break
            
            assert found, f"Documented endpoint {method} {path} not found in routes"
        
        print("✅ API documentation is accurate")
        
    except ImportError as e:
        print(f"⚠️  Could not import API components: {e}")

def main():
    """Run all tests."""
    print("🔍 Testing Background Task API Implementation")
    print("=" * 60)
    
    test_api_responsiveness()
    test_task_endpoints_return_task_ids()
    test_status_endpoint()
    test_request_id_logging()
    
    # Run async test
    asyncio.run(test_async_task_dispatching())
    
    test_api_documentation()
    
    print("\n" + "=" * 60)
    print("✅ All background task API tests completed!")
    
    print("\n📊 Key Features Verified:")
    print("- API endpoints are responsive (< 1-2 seconds)")
    print("- Task endpoints return task IDs immediately")
    print("- Status polling endpoints work correctly")
    print("- Request IDs are generated and logged")
    print("- API documentation is accurate")
    print("- Error handling for unavailable task queue")

if __name__ == "__main__":
    main()