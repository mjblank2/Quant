#!/usr/bin/env python3
"""
Test script to verify the cron module entrypoints work correctly.
"""
import subprocess
import os
import sys

def test_module_entrypoints():
    """Test that the module entrypoints work correctly with the fixed dockerCommand format."""
    
    # Set up test environment
    env = os.environ.copy()
    env['DATABASE_URL'] = 'sqlite:///test.db'
    env['SERVICE'] = 'cron'
    
    print("Testing cron module entrypoints...")
    
    # Test 1: data.universe module execution via entrypoint
    print("\n1. Testing data.universe module via entrypoint:")
    try:
        result = subprocess.run(
            ['bash', 'scripts/entrypoint.sh', 'python', '-m', 'data.universe'],
            cwd='/home/runner/work/Quant/Quant',
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"   Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ data.universe module executed successfully via entrypoint")
            
            # Check for expected log messages
            if "Universe rebuild placeholder" in result.stdout:
                print("✅ Universe rebuild function was called")
            if "Command completed with exit code: 0" in result.stdout:
                print("✅ Entrypoint reported successful completion")
        else:
            print("❌ data.universe module execution failed")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⚠️  data.universe command timed out")
    except Exception as e:
        print(f"❌ data.universe test failed: {e}")
    
    # Test 2: data.ingest module execution via entrypoint
    print("\n2. Testing data.ingest module via entrypoint:")
    try:
        result = subprocess.run(
            ['bash', 'scripts/entrypoint.sh', 'python', '-m', 'data.ingest', '--days', '7'],
            cwd='/home/runner/work/Quant/Quant',
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"   Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ data.ingest module executed successfully via entrypoint")
            
            # Check for expected log messages
            if "Starting data ingestion for 7 days" in result.stdout:
                print("✅ Data ingestion function was called with correct parameter")
            if "Command completed with exit code: 0" in result.stdout:
                print("✅ Entrypoint reported successful completion")
        else:
            print("❌ data.ingest module execution failed")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⚠️  data.ingest command timed out")
    except Exception as e:
        print(f"❌ data.ingest test failed: {e}")
    
    # Test 3: data.ingest help functionality
    print("\n3. Testing data.ingest --help via entrypoint:")
    try:
        result = subprocess.run(
            ['bash', 'scripts/entrypoint.sh', 'python', '-m', 'data.ingest', '--help'],
            cwd='/home/runner/work/Quant/Quant',
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"   Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ data.ingest --help executed successfully")
            
            # Check for expected help content
            if "Ingest market data for universe" in result.stdout:
                print("✅ Help message contains expected description")
            if "--days DAYS" in result.stdout:
                print("✅ Help message shows --days parameter")
        else:
            print("❌ data.ingest --help execution failed")
            
    except Exception as e:
        print(f"❌ data.ingest --help test failed: {e}")

    # Test 4: Direct module execution (without entrypoint) 
    print("\n4. Testing direct module execution:")
    try:
        result = subprocess.run(
            ['python', '-m', 'data.universe'],
            cwd='/home/runner/work/Quant/Quant',
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"   Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ Direct module execution works")
        else:
            print("❌ Direct module execution failed")
            
    except Exception as e:
        print(f"❌ Direct module test failed: {e}")

def test_error_handling():
    """Test that modules handle errors correctly and return proper exit codes."""
    
    print("\n\n=== Testing Error Handling ===")
    
    env = os.environ.copy()
    env['DATABASE_URL'] = 'sqlite:///test.db'
    
    # Test invalid arguments for data.ingest
    print("\n1. Testing invalid arguments:")
    try:
        result = subprocess.run(
            ['python', '-m', 'data.ingest', '--invalid-arg'],
            cwd='/home/runner/work/Quant/Quant',
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print("✅ Module correctly rejects invalid arguments")
        else:
            print("❌ Module should have failed with invalid arguments")
            
    except Exception as e:
        print(f"❌ Invalid args test failed: {e}")

if __name__ == "__main__":
    test_module_entrypoints()
    test_error_handling()
    print("\n🎉 Cron module entrypoints test completed!")