#!/usr/bin/env python3
"""
Test script to verify the Alembic migration fix.
"""

import os
import tempfile
import subprocess
import sys
from pathlib import Path

def test_migration_fix():
    """Test that the migration fix resolves the transaction issue."""
    print("🧪 Testing Alembic migration fix...")
    
    # Use a temporary SQLite database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        test_db_path = tmp.name
    
    try:
        # Set up environment
        env = os.environ.copy()
        env['DATABASE_URL'] = f'sqlite:///{test_db_path}'
        
        # Test the migration on SQLite first
        print("✅ Testing migration with SQLite...")
        result = subprocess.run(
            ['alembic', 'upgrade', 'heads'],
            capture_output=True,
            text=True,
            env=env,
            cwd='/home/runner/work/Quant/Quant'
        )
        
        if result.returncode == 0:
            print("✅ SQLite migration successful")
        else:
            print(f"❌ SQLite migration failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
        
        # Check current revision
        result = subprocess.run(
            ['alembic', 'current'],
            capture_output=True,
            text=True,
            env=env,
            cwd='/home/runner/work/Quant/Quant'
        )
        
        if result.returncode == 0:
            print(f"✅ Current revision: {result.stdout.strip()}")
        else:
            print(f"❌ Failed to get current revision: {result.stderr}")
            return False
        
        # Test downgrade to verify transaction handling
        print("✅ Testing downgrade...")
        result = subprocess.run(
            ['alembic', 'downgrade', '20250827_12'],
            capture_output=True,
            text=True,
            env=env,
            cwd='/home/runner/work/Quant/Quant'
        )
        
        if result.returncode == 0:
            print("✅ Downgrade successful")
        else:
            print(f"❌ Downgrade failed:")
            print(f"STDERR: {result.stderr}")
            return False
        
        # Test upgrade again
        print("✅ Testing upgrade again...")
        result = subprocess.run(
            ['alembic', 'upgrade', 'heads'],
            capture_output=True,
            text=True,
            env=env,
            cwd='/home/runner/work/Quant/Quant'
        )
        
        if result.returncode == 0:
            print("✅ Re-upgrade successful")
            return True
        else:
            print(f"❌ Re-upgrade failed:")
            print(f"STDERR: {result.stderr}")
            return False
            
    finally:
        # Clean up
        if os.path.exists(test_db_path):
            os.unlink(test_db_path)

if __name__ == "__main__":
    success = test_migration_fix()
    if success:
        print("\n🎉 Migration fix test passed!")
        sys.exit(0)
    else:
        print("\n❌ Migration fix test failed!")
        sys.exit(1)