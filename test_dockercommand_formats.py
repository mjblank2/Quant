#!/usr/bin/env python3
"""
Test script to verify that array-form dockerCommand fixes the exit 127 issue.
"""
import subprocess
import os
import sys

def test_dockercommand_formats():
    """Compare string vs array dockerCommand formats."""
    
    env = os.environ.copy()
    env['DATABASE_URL'] = 'sqlite:///test.db'
    env['SERVICE'] = 'cron'
    
    print("Testing dockerCommand format differences...")
    
    # Test the old problematic string format that causes exit 127
    print("\n1. Testing OLD string format (should fail with exit 127):")
    try:
        result = subprocess.run(
            ['bash', 'scripts/entrypoint.sh', 'python -m data.universe'],
            cwd='/home/runner/work/Quant/Quant',
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"   Exit code: {result.returncode}")
        
        if result.returncode == 127:
            print("✅ String format correctly fails with exit 127 (as expected)")
            print("   This demonstrates the original problem")
        else:
            print(f"⚠️  Expected exit 127, got {result.returncode}")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            
    except Exception as e:
        print(f"❌ String format test failed: {e}")
    
    # Test the new array format that should work
    print("\n2. Testing NEW array format (should succeed):")
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
            print("✅ Array format works correctly with exit 0")
            print("   This demonstrates the fix")
            
            if "Command completed with exit code: 0" in result.stdout:
                print("✅ Entrypoint properly handled array arguments")
        else:
            print(f"❌ Array format failed with exit {result.returncode}")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Array format test failed: {e}")
    
    # Test the ingest command with parameters
    print("\n3. Testing NEW ingest array format with parameters:")
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
            print("✅ Array format works with parameters")
            
            if "Starting data ingestion for 7 days" in result.stdout:
                print("✅ Parameters were correctly parsed")
        else:
            print(f"❌ Array format with parameters failed")
            
    except Exception as e:
        print(f"❌ Array format with parameters test failed: {e}")

    # Test pipeline command
    print("\n4. Testing NEW pipeline array format:")
    try:
        result = subprocess.run(
            ['bash', 'scripts/entrypoint.sh', 'python', 'run_pipeline.py'],
            cwd='/home/runner/work/Quant/Quant',
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"   Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ Pipeline array format works correctly")
        elif "Command completed with exit code:" in result.stdout:
            print("⚠️  Pipeline executed but returned non-zero (may be expected)")
        else:
            print(f"❌ Pipeline array format failed")
            
    except Exception as e:
        print(f"❌ Pipeline array format test failed: {e}")

if __name__ == "__main__":
    test_dockercommand_formats()
    print("\n🎉 dockerCommand format comparison test completed!")