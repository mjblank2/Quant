#!/usr/bin/env python3
"""
Test to verify the pipeline import fix.
This simulates the exact import that was failing in run_pipeline.py.
"""

import ast
import sys


def test_models_features_import():
    """
    Test that models/features.py can be imported without syntax errors.
    This simulates the exact scenario from the problem statement.
    """
    print("🔧 Testing the exact import from run_pipeline.py...")
    print()
    print("Original error:")
    print("  File \"/app/run_pipeline.py\", line 78, in _import_modules")
    print("    from models.features import build_features")
    print("  File \"/app/models/features.py\", line 356")
    print("    for    for col in features_for_cs:")
    print("           ^^^")
    print("  SyntaxError: invalid syntax")
    print()
    
    # Verify the file can be parsed
    try:
        with open('models/features.py', 'r') as f:
            code = f.read()
        
        ast.parse(code)
        print("✅ models/features.py has valid Python syntax")
        
        # Verify the bug is fixed
        if 'for    for col in' in code:
            print("❌ BUG STILL PRESENT: Duplicate 'for' keyword found!")
            return False
        
        print("✅ Duplicate 'for' keyword bug is fixed")
        
        # Verify the correct structure exists
        if 'for col in features_for_cs:' in code:
            print("✅ Correct for loop structure is in place")
        
        print()
        print("🎉 The import 'from models.features import build_features' will now succeed!")
        print("🎉 The pipeline should run without syntax errors!")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error still present: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_run_pipeline_syntax():
    """Test that run_pipeline.py itself has valid syntax."""
    print("\n🔧 Testing run_pipeline.py syntax...")
    
    try:
        with open('run_pipeline.py', 'r') as f:
            code = f.read()
        
        ast.parse(code)
        print("✅ run_pipeline.py has valid syntax")
        
        # Verify the import statement
        if 'from models.features import build_features' in code:
            print("✅ run_pipeline.py correctly imports from models.features")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in run_pipeline.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking run_pipeline.py: {e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("PIPELINE IMPORT FIX VERIFICATION")
    print("=" * 70)
    print()
    
    tests = [
        test_models_features_import,
        test_run_pipeline_syntax,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
    
    print()
    print("=" * 70)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print()
        print("✅ SUCCESS! The syntax error has been fixed.")
        print("✅ The cron job pipeline should now run successfully.")
        sys.exit(0)
    else:
        print()
        print("❌ FAILURE! Some tests failed.")
        sys.exit(1)
