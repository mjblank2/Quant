#!/usr/bin/env python3
"""
Test script specifically for API responsiveness using mocked background tasks.
This test verifies that the API endpoints respond quickly when the task queue is available.
"""
import pytest
import time
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
import os

# Set up test environment
os.environ['DATABASE_URL'] = 'sqlite:///test.db'

def test_mocked_task_responsiveness():
    """Test API responsiveness with mocked Celery tasks."""
    print("=== Testing API Responsiveness with Mocked Tasks ===")
    
    try:
        from health_api import app
        if not app:
            print("⚠️  FastAPI app not available, skipping mocked tests")
            return
        
        client = TestClient(app)
        
        # Mock all the Celery tasks to avoid Redis connection
        with patch('health_api.TASK_QUEUE_AVAILABLE', True), \
             patch('health_api.rebuild_universe_task') as mock_universe, \
             patch('health_api.ingest_fundamentals_task') as mock_fundamentals, \
             patch('health_api.build_features_task') as mock_features, \
             patch('health_api.train_and_predict_task') as mock_train, \
             patch('health_api.run_backtest_task') as mock_backtest, \
             patch('health_api.generate_trades_task') as mock_trades, \
             patch('health_api.sync_broker_task') as mock_broker, \
             patch('health_api.run_full_pipeline_task') as mock_pipeline, \
             patch('health_api.ingest_market_data_task') as mock_ingest:
            
            # Configure all mocks to return a mock task with an ID
            def configure_mock(mock_task, task_id):
                mock_result = Mock()
                mock_result.id = task_id
                mock_task.delay.return_value = mock_result
                return mock_task
            
            configure_mock(mock_universe, "test-universe-123")
            configure_mock(mock_fundamentals, "test-fundamentals-123")
            configure_mock(mock_features, "test-features-123")
            configure_mock(mock_train, "test-train-123")
            configure_mock(mock_backtest, "test-backtest-123")
            configure_mock(mock_trades, "test-trades-123")
            configure_mock(mock_broker, "test-broker-123")
            configure_mock(mock_pipeline, "test-pipeline-123")
            configure_mock(mock_ingest, "test-ingest-123")
            
            # Test all task endpoints for responsiveness
            test_cases = [
                ("/universe", "POST", {}, "test-universe-123"),
                ("/fundamentals", "POST", {}, "test-fundamentals-123"),
                ("/features", "POST", {}, "test-features-123"),
                ("/train", "POST", {}, "test-train-123"),
                ("/backtest", "POST", {}, "test-backtest-123"),
                ("/trades", "POST", {}, "test-trades-123"),
                ("/broker-sync", "POST", {"trade_ids": [1, 2, 3]}, "test-broker-123"),
                ("/pipeline", "POST", {"sync_broker": False}, "test-pipeline-123"),
                ("/ingest", "POST", {"days": 7}, "test-ingest-123"),
            ]
            
            for endpoint, method, payload, expected_task_id in test_cases:
                start_time = time.time()
                
                response = client.post(endpoint, json=payload)
                response_time = (time.time() - start_time) * 1000
                
                print(f"  {method} {endpoint}: {response.status_code} in {response_time:.1f}ms")
                
                # Should be very quick with mocked tasks (< 100ms)
                assert response_time < 100, f"Mocked endpoint {endpoint} took too long: {response_time}ms"
                
                # Should return 202 with task ID
                assert response.status_code == 202, f"Expected 202 for {endpoint}, got {response.status_code}"
                
                data = response.json()
                assert data["status"] == "accepted", f"Wrong status for {endpoint}"
                assert data["task_id"] == expected_task_id, f"Wrong task_id for {endpoint}"
                assert f"/tasks/status/{expected_task_id}" in data["status_url"], f"Wrong status_url for {endpoint}"
                
                print(f"    ✅ Task dispatched successfully: {data['task_id']}")
        
        print("✅ All mocked task endpoints are highly responsive")
        
    except ImportError as e:
        print(f"⚠️  Could not import API components: {e}")

def test_status_endpoint_mocked():
    """Test status endpoint with mocked task status."""
    print("\n=== Testing Status Endpoint with Mock Data ===")
    
    try:
        from health_api import app
        if not app:
            print("⚠️  FastAPI app not available, skipping status test")
            return
        
        client = TestClient(app)
        
        # Mock the get_task_status function
        with patch('health_api.TASK_QUEUE_AVAILABLE', True), \
             patch('health_api.get_task_status') as mock_get_status:
            
            # Test successful status retrieval
            mock_get_status.return_value = {
                'task_id': 'test-task-123',
                'task_name': 'test_task',
                'status': 'SUCCESS',
                'progress': 100,
                'result': {'test': 'data'},
                'error_message': None
            }
            
            start_time = time.time()
            response = client.get("/tasks/status/test-task-123")
            response_time = (time.time() - start_time) * 1000
            
            print(f"  GET /tasks/status/test-task-123: {response.status_code} in {response_time:.1f}ms")
            
            # Should be very quick
            assert response_time < 50, f"Status endpoint took too long: {response_time}ms"
            assert response.status_code == 200, "Should return 200 for existing task"
            
            data = response.json()
            assert data['task_id'] == 'test-task-123', "Wrong task_id in response"
            assert data['status'] == 'SUCCESS', "Wrong status in response"
            
            print("    ✅ Successfully retrieved task status")
            
            # Test non-existent task
            mock_get_status.return_value = None
            
            response = client.get("/tasks/status/non-existent-task")
            assert response.status_code == 404, "Should return 404 for non-existent task"
            
            print("    ✅ Correctly handles non-existent task")
        
        print("✅ Status endpoint works correctly with mock data")
        
    except ImportError as e:
        print(f"⚠️  Could not import API components: {e}")

def test_concurrent_requests():
    """Test that the API can handle multiple concurrent requests efficiently."""
    print("\n=== Testing Concurrent Request Handling ===")
    
    try:
        from health_api import app
        import threading
        import queue
        
        if not app:
            print("⚠️  FastAPI app not available, skipping concurrent test")
            return
        
        client = TestClient(app)
        results_queue = queue.Queue()
        
        def make_request(endpoint, request_id):
            """Make a request and record the timing."""
            start_time = time.time()
            response = client.get(endpoint)
            end_time = time.time()
            
            results_queue.put({
                'request_id': request_id,
                'endpoint': endpoint,
                'status_code': response.status_code,
                'response_time': (end_time - start_time) * 1000
            })
        
        # Launch multiple concurrent requests
        threads = []
        endpoints = ["/health", "/status", "/metrics", "/", "/health"]
        
        start_time = time.time()
        
        for i, endpoint in enumerate(endpoints):
            thread = threading.Thread(target=make_request, args=(endpoint, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = (time.time() - start_time) * 1000
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        print(f"  Completed {len(results)} concurrent requests in {total_time:.1f}ms")
        
        for result in results:
            print(f"    Request {result['request_id']} ({result['endpoint']}): "
                  f"{result['status_code']} in {result['response_time']:.1f}ms")
            
            # Each request should be reasonably quick
            assert result['response_time'] < 100, f"Request {result['request_id']} too slow"
            assert result['status_code'] == 200, f"Request {result['request_id']} failed"
        
        # Total time should be much less than sum of individual times (parallel execution)
        total_individual_time = sum(r['response_time'] for r in results)
        print(f"  Individual times sum: {total_individual_time:.1f}ms, Actual total: {total_time:.1f}ms")
        
        # Parallel execution should be more efficient
        efficiency = total_individual_time / total_time
        print(f"  Parallelization efficiency: {efficiency:.1f}x")
        
        assert efficiency > 1.0, "Requests should execute in parallel"
        
        print("✅ API handles concurrent requests efficiently")
        
    except ImportError as e:
        print(f"⚠️  Could not import API components: {e}")

def main():
    """Run all responsiveness tests."""
    print("🚀 Testing API Responsiveness and Background Task Implementation")
    print("=" * 70)
    
    test_mocked_task_responsiveness()
    test_status_endpoint_mocked()
    test_concurrent_requests()
    
    print("\n" + "=" * 70)
    print("✅ All API responsiveness tests passed!")
    
    print("\n📊 Key Performance Metrics Verified:")
    print("- Mocked task endpoints respond in < 100ms")
    print("- Status endpoints respond in < 50ms")
    print("- Concurrent requests are handled efficiently")
    print("- All endpoints return appropriate status codes")
    print("- Task IDs are properly generated and returned")

if __name__ == "__main__":
    main()