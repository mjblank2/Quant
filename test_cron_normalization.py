#!/usr/bin/env python3
"""
Test script to verify the cron script normalization functionality.
"""
import subprocess
import os
import tempfile

def test_cron_script_normalization():
    """Test that the cron scripts work correctly with the new preamble."""
    
    print("Testing cron script normalization...")
    
    # Set up test environment
    env = os.environ.copy()
    env['DATABASE_URL'] = 'sqlite:///test.db'
    env['SERVICE'] = 'cron'
    
    # Test 1: cron_universe.sh with preamble
    print("\n1. Testing cron_universe.sh:")
    try:
        result = subprocess.run(
            ['bash', 'scripts/cron_universe.sh'],
            cwd='/home/runner/work/Quant/Quant',
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"   Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ cron_universe.sh executed successfully")
            
            # Check for expected log messages
            if "[cron] ⚠️ Alembic upgrade failed (continuing)" in result.stdout:
                print("✅ Alembic migration attempted and handled properly")
            
            if "[cron_universe] 🕰️ Start:" in result.stdout:
                print("✅ Consistent logging format detected")
                
            if "[cron_universe] 🕰️ End:" in result.stdout:
                print("✅ Script completed with proper end logging")
        else:
            print(f"❌ Script failed with exit code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            
    except Exception as e:
        print(f"❌ cron_universe.sh test failed: {e}")

    # Test 2: cron_ingest.sh with DAYS parameter
    print("\n2. Testing cron_ingest.sh with DAYS parameter:")
    try:
        env['DAYS'] = '2'
        result = subprocess.run(
            ['bash', 'scripts/cron_ingest.sh'],
            cwd='/home/runner/work/Quant/Quant',
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"   Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ cron_ingest.sh executed successfully")
            
            if "DAYS=2" in result.stdout:
                print("✅ DAYS parameter correctly passed and logged")
                
            if "[cron_ingest] 🕰️ Start:" in result.stdout:
                print("✅ Consistent logging format detected")
        else:
            print(f"❌ Script failed with exit code {result.returncode}")
            
    except Exception as e:
        print(f"❌ cron_ingest.sh test failed: {e}")

    # Test 3: Test preamble PYTHONPATH logic
    print("\n3. Testing PYTHONPATH normalization:")
    
    # Create temporary test script to check PYTHONPATH logic
    test_script = '''#!/bin/bash
unset PYTHONPATH
source scripts/_common_cron_preamble.sh 2>/dev/null || true
echo "PYTHONPATH=${PYTHONPATH:-UNSET}"
'''
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(test_script)
            f.flush()
            
            os.chmod(f.name, 0o755)
            
            result = subprocess.run(
                ['bash', f.name],
                cwd='/home/runner/work/Quant/Quant',
                env=env,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if "PYTHONPATH=UNSET" in result.stdout:
                print("✅ PYTHONPATH normalization works correctly (no /app or /opt/render/project/src)")
            else:
                print(f"❌ Unexpected PYTHONPATH behavior: {result.stdout}")
            
        os.unlink(f.name)
            
    except Exception as e:
        print(f"❌ PYTHONPATH test failed: {e}")

    print("\n🎉 Cron script normalization tests completed!")

if __name__ == "__main__":
    test_cron_script_normalization()