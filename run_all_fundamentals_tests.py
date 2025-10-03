#!/usr/bin/env python3
"""
Comprehensive test suite for the fundamentals upsert fix.
Runs all validation tests to ensure the production issue is resolved.
"""

import sys
import subprocess


def run_test(script_name, description):
    """Run a test script and return success status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"{'='*70}")
    
    result = subprocess.run(
        ["python", script_name],
        capture_output=True,
        text=True,
        cwd="/home/runner/work/Quant/Quant"
    )
    
    print(result.stdout)
    if result.stderr and "timestamp" not in result.stderr:  # Filter out log messages
        print(result.stderr)
    
    return result.returncode == 0


def main():
    """Run all validation tests."""
    print("="*70)
    print("FUNDAMENTALS UPSERT FIX - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("test_fundamentals_upsert_fix.py", "Unit Tests - Deduplication Logic"),
        ("test_fundamentals_integration.py", "Integration Tests - Duplicate Key Scenarios"),
        ("demo_fundamentals_fix.py", "Manual Validation Demo"),
    ]
    
    results = []
    for script, desc in tests:
        success = run_test(script, desc)
        results.append((desc, success))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for desc, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {desc}")
        if not success:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nThe fundamentals upsert fix successfully resolves:")
        print("  • UniqueViolation errors for duplicate (symbol, as_of) keys")
        print("  • Within-batch duplicate handling")
        print("  • Updates to existing records via ON CONFLICT DO UPDATE")
        print("\nThe EOD pipeline should now run without errors.")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Please review the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
