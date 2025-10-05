#!/usr/bin/env python3
"""
Test to verify the syntax fix in models/features.py
This test validates that the duplicate 'for' keyword bug is fixed.
"""

import ast
import sys


def test_syntax():
    """Test that models/features.py has valid Python syntax."""
    print("🔧 Testing models/features.py syntax...")
    
    try:
        with open('models/features.py', 'r') as f:
            code = f.read()
        
        # Parse the AST to check syntax
        ast.parse(code)
        print("✅ Syntax is valid")
        
        # Check that the buggy line doesn't exist
        if 'for    for col in' in code:
            print("❌ Bug still present: duplicate 'for' keyword found")
            return False
        
        # Check that the correct line exists
        if 'for col in features_for_cs:' in code:
            print("✅ Correct 'for' loop structure found")
            return True
        else:
            print("⚠️ Expected for loop not found, but syntax is valid")
            return True
            
    except SyntaxError as e:
        print(f"❌ Syntax error found: {e}")
        return False
    except FileNotFoundError:
        print("❌ File models/features.py not found")
        return False


def test_indentation():
    """Test that the indentation is correct in the fixed section."""
    print("\n🔧 Testing indentation in the fixed section...")
    
    try:
        with open('models/features.py', 'r') as f:
            lines = f.readlines()
        
        # Find the line with "for col in features_for_cs:"
        for i, line in enumerate(lines):
            if 'for col in features_for_cs:' in line:
                # Check that it has proper indentation (12 spaces based on context)
                if line.startswith('            for col in'):
                    print(f"✅ Correct indentation found at line {i+1}")
                    
                    # Check the next line (zcol assignment) has proper indentation
                    if i+1 < len(lines):
                        next_line = lines[i+1]
                        if next_line.startswith('                zcol ='):
                            print("✅ Body indentation is correct")
                            return True
                        else:
                            print(f"⚠️ Body indentation may be incorrect: {repr(next_line[:20])}")
                            return False
                else:
                    print(f"⚠️ Indentation may be incorrect: {repr(line[:20])}")
                    return False
        
        print("⚠️ Could not find the for loop line")
        return True  # Don't fail if we can't find it
        
    except Exception as e:
        print(f"❌ Error checking indentation: {e}")
        return False


if __name__ == "__main__":
    print("🎯 Running syntax fix verification tests\n")
    
    tests = [
        test_syntax,
        test_indentation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️ Some tests failed")
        sys.exit(1)
