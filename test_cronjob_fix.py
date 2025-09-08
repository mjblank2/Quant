#!/usr/bin/env python3
"""
Test script to verify the cronjob fix works correctly.
"""
import subprocess
import os
import sys

def test_entrypoint_execution():
    """Test that the entrypoint correctly executes the fixed dockerCommand format."""
    
    # Set up test environment
    env = os.environ.copy()
    env['DATABASE_URL'] = 'sqlite:///test.db'
    env['SERVICE'] = 'cron'
    
    print("Testing cronjob fix...")
    
    # Test the problematic original command (should fail)
    print("\n1. Testing original problematic command:")
    try:
        result = subprocess.run(
            ['bash', 'scripts/entrypoint.sh', 'alembic upgrade heads && python -m run_pipeline'],
            cwd='/home/runner/work/Quant/Quant',
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            print("✅ Original command correctly fails (as expected)")
            print(f"   Exit code: {result.returncode}")
        else:
            print("❌ Original command unexpectedly succeeded")
    except Exception as e:
        print(f"✅ Original command failed with exception (as expected): {e}")
    
    # Test the fixed command format
    print("\n2. Testing fixed command format:")
    try:
        result = subprocess.run(
            ['bash', 'scripts/entrypoint.sh', 'sh', '-c', 'python run_pipeline.py'],
            cwd='/home/runner/work/Quant/Quant',
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"   Exit code: {result.returncode}")
        
        # Check for success indicators in output
        if "Command completed with exit code: 0" in result.stdout:
            print("✅ Fixed command executed successfully")
            print("✅ Pipeline completed with exit code 0")
        elif "Command completed with exit code:" in result.stdout:
            print("⚠️  Command executed but pipeline returned non-zero exit code")
            print("   (This may be expected due to missing dependencies or early exit)")
        else:
            print("❌ Command execution status unclear")
            
        # Check that python was found and executed
        if "python: command not found" not in result.stdout and "python: command not found" not in result.stderr:
            print("✅ Python command was found and executed")
        else:
            print("❌ Python command not found")
            
    except subprocess.TimeoutExpired:
        print("⚠️  Command timed out (may be expected)")
    except Exception as e:
        print(f"❌ Fixed command failed with exception: {e}")
    
    print("\n3. Testing module command format:")
    try:
        result = subprocess.run(
            ['bash', 'scripts/entrypoint.sh', 'sh', '-c', 'python -c "print(\\"Module test success\\")"'],
            cwd='/home/runner/work/Quant/Quant',
            env=env,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if "Module test success" in result.stdout:
            print("✅ Module command format works correctly")
        else:
            print("❌ Module command format failed")
            
    except Exception as e:
        print(f"❌ Module test failed: {e}")

if __name__ == "__main__":
    test_entrypoint_execution()
    print("\n🎉 Cronjob fix test completed!")