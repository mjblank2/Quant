#!/usr/bin/env python3
"""
Demo script showing how to use the new background task API endpoints.
This demonstrates the proper way to dispatch long-running tasks and monitor their progress.
"""
import requests
import time
import json
import requests
import time


def demo_api_usage(base_url="http://localhost:8000"):
    """
    Demonstrate how to use the background task API endpoints.
    
    Args:
        base_url: Base URL of the FastAPI application
    """
    print("🚀 Background Task API Demo")
    print("=" * 50)
    
    # Check API health
    print("1. Checking API health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ API is healthy")
        else:
            print(f"   ⚠️  API health check returned {response.status_code}")
    except requests.RequestException as e:
        print(f"   ❌ Failed to connect to API: {e}")
        print("   💡 Make sure to start the FastAPI server:")
        print("      uvicorn health_api:app --host 0.0.0.0 --port 8000")
        return
    
    # Get API documentation
    print("\n2. Getting API documentation...")
    response = requests.get(f"{base_url}/")
    if response.status_code == 200:
        docs = response.json()
        print("   Available endpoints:")
        for name, endpoint in docs.get("endpoints", {}).items():
            print(f"     - {name}: {endpoint}")
    
    # Demonstrate task dispatching
    print("\n3. Dispatching background tasks...")
    
    task_endpoints = [
        ("/universe", "Universe rebuild", {}),
        ("/ingest", "Data ingestion", {"days": 7}),
        ("/features", "Feature building", {}),
        ("/train", "Model training", {}),
    ]
    
    dispatched_tasks = []
    
    for endpoint, description, payload in task_endpoints:
        print(f"\n   Dispatching {description}...")
        try:
            start_time = time.time()
            response = requests.post(f"{base_url}{endpoint}", json=payload, timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            print(f"   Response time: {response_time:.1f}ms")
            
            if response.status_code == 202:
                data = response.json()
                task_id = data["task_id"]
                status_url = data["status_url"]
                dispatched_tasks.append((task_id, description, status_url))
                print(f"   ✅ Task dispatched successfully!")
                print(f"      Task ID: {task_id}")
                print(f"      Status URL: {status_url}")
                
            elif response.status_code == 503:
                print(f"   ⚠️  Service unavailable: {response.json()['detail']}")
                
            else:
                print(f"   ❌ Unexpected response: {response.status_code}")
                print(f"      {response.text}")
                
        except requests.RequestException as e:
            print(f"   ❌ Request failed: {e}")
    
    # Monitor task progress
    if dispatched_tasks:
        print("\n4. Monitoring task progress...")
        
        for task_id, description, status_url in dispatched_tasks:
            print(f"\n   Checking status of {description} (ID: {task_id[:8]}...)...")
            
            try:
                response = requests.get(f"{base_url}{status_url}", timeout=5)
                
                if response.status_code == 200:
                    status_data = response.json()
                    print(f"   ✅ Task found:")
                    print(f"      Status: {status_data.get('status', 'unknown')}")
                    print(f"      Progress: {status_data.get('progress', 0)}%")
                    if status_data.get('error_message'):
                        print(f"      Error: {status_data['error_message']}")
                    if status_data.get('result'):
                        print(f"      Result: {status_data['result']}")
                        
                elif response.status_code == 404:
                    print(f"   ⚠️  Task not found (may not be persisted yet)")
                    
                elif response.status_code == 503:
                    print(f"   ⚠️  Task queue unavailable")
                    
                else:
                    print(f"   ❌ Unexpected status: {response.status_code}")
                    
            except requests.RequestException as e:
                print(f"   ❌ Status check failed: {e}")
    
    # Demonstrate metrics endpoint
    print("\n5. Checking API metrics...")
    try:
        response = requests.get(f"{base_url}/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.json()
            print("   📊 API Metrics:")
            print(f"      Total requests: {metrics.get('total_requests', 0)}")
            print(f"      Recent requests (5min): {metrics.get('recent_requests_5min', 0)}")
            print(f"      Avg response time: {metrics.get('avg_response_time_ms', 0):.1f}ms")
        else:
            print(f"   ⚠️  Metrics unavailable: {response.status_code}")
    except requests.RequestException as e:
        print(f"   ❌ Metrics check failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Demo completed!")
    print("\n📋 Key Takeaways:")
    print("- All task endpoints return immediately with task IDs")
    print("- Use /tasks/status/{task_id} to monitor progress")
    print("- API handles task queue unavailability gracefully")
    print("- Request IDs are automatically generated for tracing")
    print("- Metrics are collected for monitoring and observability")

def main():
    """Run the demo."""
    print("💡 This demo shows how to use the background task API.")
    print("   Make sure to start the API server first:")
    print("   uvicorn health_api:app --host 0.0.0.0 --port 8000")
    print()
    
    # Try to demo with default URL
    demo_api_usage()

if __name__ == "__main__":
    main()