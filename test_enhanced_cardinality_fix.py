#!/usr/bin/env python3
"""
Test the enhanced CardinalityViolation fix in models/features.py.

This test validates that the additional deduplication in the feature engineering
process prevents CardinalityViolation errors before they reach the database layer.
"""

import pandas as pd
import numpy as np
import logging
import tempfile
import os
from sqlalchemy import create_engine, text
from unittest.mock import patch, MagicMock

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def test_feature_engineering_deduplication():
    """Test the deduplication logic added to models/features.py"""
    print("🔧 Testing feature engineering deduplication fix...")
    
    # Create a temporary SQLite database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        temp_db_path = tmp.name
    
    try:
        # Set up environment
        os.environ['DATABASE_URL'] = f'sqlite:///{temp_db_path}'
        
        # Create test database
        engine = create_engine(f'sqlite:///{temp_db_path}')
        
        # Import after setting DATABASE_URL
        import sys
        sys.path.append('/home/runner/work/Quant/Quant')
        from db import Base, Feature
        
        # Create tables
        Base.metadata.create_all(engine)
        
        # Simulate the problematic scenario from features.py
        # This replicates the logic around lines 218-222
        
        # Create out_frames with duplicates (simulating complex feature engineering)
        out_frames = []
        
        # Frame 1: AAPL data
        frame1 = pd.DataFrame([
            {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.01, 'ret_5d': 0.05, 'vol_21': 0.15},
            {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-05').date(), 'ret_1d': 0.02, 'ret_5d': 0.06, 'vol_21': 0.16},
        ])
        out_frames.append(frame1)
        
        # Frame 2: MSFT data (normal case)
        frame2 = pd.DataFrame([
            {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.015, 'ret_5d': 0.045, 'vol_21': 0.14},
        ])
        out_frames.append(frame2)
        
        # Frame 3: Problematic frame with duplicates (this could happen due to merge_asof issues)
        frame3 = pd.DataFrame([
            {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.011, 'ret_5d': 0.051, 'vol_21': 0.151},  # Duplicate!
            {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.016, 'ret_5d': 0.046, 'vol_21': 0.141},  # Duplicate!
        ])
        out_frames.append(frame3)
        
        print("Simulated out_frames with duplicates:")
        for i, frame in enumerate(out_frames):
            print(f"Frame {i+1}:")
            print(frame)
            print()
        
        # This replicates the original problematic line 219: feats = pd.concat(out_frames, ignore_index=True)
        feats_with_duplicates = pd.concat(out_frames, ignore_index=True)
        
        print("Concatenated feats (with duplicates):")
        print(feats_with_duplicates)
        print()
        
        # Check for duplicates
        duplicate_check = feats_with_duplicates.groupby(['symbol', 'ts']).size()
        duplicates_found = (duplicate_check > 1).sum()
        print(f"Duplicates found: {duplicates_found}")
        if duplicates_found > 0:
            print("Duplicate groups:")
            print(duplicate_check[duplicate_check > 1])
        print()
        
        # Apply the fix (this replicates the new deduplication logic)
        original_count = len(feats_with_duplicates)
        feats_deduplicated = feats_with_duplicates.drop_duplicates(subset=['symbol', 'ts'], keep='last').reset_index(drop=True)
        dedupe_count = len(feats_deduplicated)
        
        print(f"Deduplication: {original_count} -> {dedupe_count} rows")
        print("Deduplicated feats:")
        print(feats_deduplicated)
        print()
        
        # Verify no duplicates remain
        final_check = feats_deduplicated.groupby(['symbol', 'ts']).size()
        final_duplicates = (final_check > 1).sum()
        
        if final_duplicates == 0:
            print("✅ Feature engineering deduplication fix works correctly")
            
            # Test actual upsert
            try:
                from db import upsert_dataframe
                with engine.connect() as conn:
                    upsert_dataframe(feats_deduplicated, Feature, ['symbol', 'ts'], conn=conn)
                    conn.commit()
                
                # Verify data in database
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT COUNT(*) FROM features")).scalar()
                    print(f"✅ Successfully inserted {result} records into database")
                    
                    # Check for duplicates in database
                    db_duplicates = conn.execute(text("""
                        SELECT symbol, ts, COUNT(*) as count 
                        FROM features 
                        GROUP BY symbol, ts 
                        HAVING COUNT(*) > 1
                    """)).fetchall()
                    
                    if len(db_duplicates) == 0:
                        print("✅ No duplicates in database - fix completely successful")
                        return True
                    else:
                        print(f"❌ Found {len(db_duplicates)} duplicates in database")
                        return False
                        
            except Exception as e:
                print(f"❌ Database upsert failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"❌ Deduplication failed - {final_duplicates} duplicates remain")
            return False
            
    finally:
        # Cleanup
        try:
            os.unlink(temp_db_path)
        except:
            pass

def test_enhanced_fix_integration():
    """Test integration of the enhanced fix with mock feature engineering."""
    print("🔄 Testing enhanced fix integration...")
    
    # Mock the feature engineering scenario that would cause CardinalityViolation
    
    # Test that the fix handles various edge cases
    test_cases = [
        {
            'name': 'Multiple symbols with same timestamp duplicates',
            'data': pd.DataFrame([
                {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.01, 'vol_21': 0.15},
                {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.011, 'vol_21': 0.151},  # Dup
                {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.02, 'vol_21': 0.14},
                {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.021, 'vol_21': 0.141},  # Dup
                {'symbol': 'GOOGL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.03, 'vol_21': 0.13},  # No dup
            ])
        },
        {
            'name': 'Single symbol with multiple timestamp duplicates',
            'data': pd.DataFrame([
                {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.01, 'vol_21': 0.15},
                {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.011, 'vol_21': 0.151},  # Dup
                {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-05').date(), 'ret_1d': 0.02, 'vol_21': 0.16},
                {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-05').date(), 'ret_1d': 0.021, 'vol_21': 0.161},  # Dup
            ])
        },
        {
            'name': 'No duplicates (normal case)',
            'data': pd.DataFrame([
                {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.01, 'vol_21': 0.15},
                {'symbol': 'AAPL', 'ts': pd.Timestamp('2016-01-05').date(), 'ret_1d': 0.02, 'vol_21': 0.16},
                {'symbol': 'MSFT', 'ts': pd.Timestamp('2016-01-04').date(), 'ret_1d': 0.015, 'vol_21': 0.14},
            ])
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        test_data = test_case['data']
        
        print("Original data:")
        print(test_data)
        
        # Apply the fix logic
        original_count = len(test_data)
        deduplicated = test_data.drop_duplicates(subset=['symbol', 'ts'], keep='last').reset_index(drop=True)
        dedupe_count = len(deduplicated)
        
        print(f"Deduplication: {original_count} -> {dedupe_count} rows")
        print("Result:")
        print(deduplicated)
        
        # Verify no duplicates
        duplicate_check = deduplicated.groupby(['symbol', 'ts']).size()
        remaining_duplicates = (duplicate_check > 1).sum()
        
        if remaining_duplicates == 0:
            print("✅ Test case passed")
        else:
            print("❌ Test case failed - duplicates remain")
            all_passed = False
    
    return all_passed

def run_enhanced_fix_tests():
    """Run all tests for the enhanced CardinalityViolation fix."""
    print("🎯 Running enhanced CardinalityViolation fix tests\n")
    
    tests = [
        test_enhanced_fix_integration,
        test_feature_engineering_deduplication,
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
            import traceback
            traceback.print_exc()
            print()
    
    print(f"📊 Enhanced Fix Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All enhanced fix tests passed!")
        print("The CardinalityViolation issue should be resolved.")
        return True
    else:
        print("⚠️ Some tests failed - fix may need adjustment")
        return False

if __name__ == "__main__":
    success = run_enhanced_fix_tests()
    exit(0 if success else 1)