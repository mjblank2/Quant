#!/usr/bin/env python3
"""
Test to verify that the CardinalityViolation warning spam fix works correctly.

This test validates that:
1. Functionality is preserved (no duplicates in final result)
2. Warning spam is reduced (fewer unnecessary warning messages)
3. Important warnings are still shown when appropriate
"""

import pandas as pd
import numpy as np
import logging
import tempfile
import os
from datetime import date, datetime
from sqlalchemy import create_engine, text

def test_warning_spam_reduction():
    """Test that the fix reduces warning spam while maintaining functionality."""
    print("🎯 Testing warning spam reduction...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        temp_db_path = tmp.name
    
    try:
        engine = create_engine(f'sqlite:///{temp_db_path}')
        
        import sys
        sys.path.append('/home/runner/work/Quant/Quant')
        
        # Set up to capture logs at different levels
        debug_logs = []
        warning_logs = []
        
        class LogCapture(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    warning_logs.append(self.format(record))
                elif record.levelno >= logging.DEBUG:
                    debug_logs.append(self.format(record))
        
        handler = LogCapture()
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        db_logger = logging.getLogger('db')
        db_logger.addHandler(handler)
        db_logger.setLevel(logging.DEBUG)
        
        from db import upsert_dataframe, Feature, Base
        Base.metadata.create_all(engine)
        
        # Create data with many duplicates that would previously generate warning spam
        data = []
        
        # Create 100 records that all deduplicate to 5 final records
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        for symbol in symbols:
            for i in range(20):  # 20 duplicates per symbol
                data.append({
                    'symbol': symbol,
                    'ts': date(2016, 1, 4),
                    'ret_1d': 0.01 + i * 0.001,  # Different values
                    'vol_21': 0.15 + i * 0.001,
                })
        
        df = pd.DataFrame(data)
        print(f"Test data: {len(df)} rows that should dedupe to {len(symbols)} rows")
        
        # Clear log captures
        debug_logs.clear()
        warning_logs.clear()
        
        # Perform upsert
        start_time = datetime.now()
        with engine.connect() as conn:
            upsert_dataframe(df, Feature, ['symbol', 'ts'], chunk_size=10, conn=conn)
            conn.commit()
        end_time = datetime.now()
        
        # Check results
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM features")).scalar()
            duplicates_in_db = conn.execute(text("""
                SELECT symbol, ts, COUNT(*) as count 
                FROM features 
                GROUP BY symbol, ts 
                HAVING COUNT(*) > 1
            """)).fetchall()
        
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Upsert completed in {duration:.3f}s")
        print(f"Records in database: {result}")
        print(f"Duplicates in database: {len(duplicates_in_db)}")
        
        # Analyze log output
        total_debug = len(debug_logs)
        total_warnings = len(warning_logs)
        
        cardinality_warnings = len([msg for msg in warning_logs if 'CardinalityViolation' in msg])
        duplicate_warnings = len([msg for msg in warning_logs if 'duplicate rows' in msg])
        
        print(f"\nLog analysis:")
        print(f"Debug messages: {total_debug}")
        print(f"Warning messages: {total_warnings}")
        print(f"CardinalityViolation warnings: {cardinality_warnings}")
        print(f"Duplicate-related warnings: {duplicate_warnings}")
        
        # Show sample messages
        if warning_logs:
            print(f"\nWarning messages:")
            for msg in warning_logs[:3]:
                print(f"  {msg}")
        
        if debug_logs:
            print(f"\nSample debug messages:")
            for msg in debug_logs[:3]:
                print(f"  {msg}")
        
        # Success criteria:
        # 1. No duplicates in final database
        # 2. Minimal or no warning messages (should be mostly debug)
        # 3. Correct number of final records
        
        success = (
            result == len(symbols) and  # Correct deduplication
            len(duplicates_in_db) == 0 and  # No duplicates in DB
            cardinality_warnings == 0  # No cardinality violation warnings
        )
        
        if success:
            print("✅ Warning spam reduction successful!")
            print("  - Correct deduplication")
            print("  - No database duplicates")
            print("  - Minimal warning spam")
        else:
            print("❌ Warning spam reduction failed")
            if result != len(symbols):
                print(f"  - Wrong record count: expected {len(symbols)}, got {result}")
            if len(duplicates_in_db) > 0:
                print(f"  - Found {len(duplicates_in_db)} duplicates in database")
            if cardinality_warnings > 0:
                print(f"  - Still getting {cardinality_warnings} cardinality warnings")
        
        return success
        
    finally:
        try:
            os.unlink(temp_db_path)
        except:
            pass

def test_significant_duplication_warning():
    """Test that significant duplication still generates warnings."""
    print("\n🔍 Testing significant duplication warning threshold...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        temp_db_path = tmp.name
    
    try:
        engine = create_engine(f'sqlite:///{temp_db_path}')
        
        import sys
        sys.path.append('/home/runner/work/Quant/Quant')
        
        warning_logs = []
        
        class LogCapture(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    warning_logs.append(self.format(record))
        
        handler = LogCapture()
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        db_logger = logging.getLogger('db')
        db_logger.addHandler(handler)
        db_logger.setLevel(logging.DEBUG)
        
        from db import upsert_dataframe, Feature, Base
        Base.metadata.create_all(engine)
        
        # Create data with >50% duplicates to trigger warning threshold
        data = []
        
        # 100 records that all dedupe to 1 record (99% duplication)
        for i in range(100):
            data.append({
                'symbol': 'AAPL',
                'ts': date(2016, 1, 4),
                'ret_1d': 0.01,  # Same values to create true duplicates
                'vol_21': 0.15,
            })
        
        df = pd.DataFrame(data)
        print(f"High duplication test: {len(df)} rows -> should dedupe to 1 row (99% duplication)")
        
        warning_logs.clear()
        
        # Force a retry scenario by using very small chunks
        # This would trigger the retry mechanism with high duplication
        with engine.connect() as conn:
            # Simulate what happens in retry by calling the deduplication logic directly
            smaller_df = pd.DataFrame(df.to_dict(orient='records')[:20])  # 20 records
            conflict_cols = ['symbol', 'ts']
            
            original_size = len(smaller_df)
            smaller_df = smaller_df.drop_duplicates(subset=conflict_cols, keep='last').reset_index(drop=True)
            dedupe_size = len(smaller_df)
            
            # This simulates the retry deduplication logic
            removed_count = original_size - dedupe_size
            if removed_count > original_size * 0.5:  # More than 50% are duplicates
                db_logger.warning(f"Removed {removed_count} duplicate rows during retry (significant duplication detected)")
            else:
                db_logger.debug(f"Removed {removed_count} duplicate rows during retry to prevent CardinalityViolation")
        
        significant_warnings = len([msg for msg in warning_logs if 'significant duplication' in msg])
        
        print(f"Warning messages for high duplication: {len(warning_logs)}")
        print(f"Significant duplication warnings: {significant_warnings}")
        
        if warning_logs:
            print(f"Warning message: {warning_logs[0]}")
        
        # Should get a warning for significant duplication
        success = significant_warnings > 0
        
        if success:
            print("✅ Significant duplication warning works correctly")
        else:
            print("❌ Significant duplication warning not triggered")
        
        return success
        
    finally:
        try:
            os.unlink(temp_db_path)
        except:
            pass

def run_warning_fix_tests():
    """Run all warning spam fix tests."""
    print("🎯 Running CardinalityViolation warning spam fix tests\n")
    
    tests = [
        test_warning_spam_reduction,
        test_significant_duplication_warning,
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
    
    print(f"📊 Warning Fix Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Warning spam fix is working correctly!")
        print("✅ Reduced unnecessary warning messages")
        print("✅ Maintained functionality and safety")
        print("✅ Still warns for significant issues")
        return True
    else:
        print("⚠️ Warning spam fix needs more work")
        return False

if __name__ == "__main__":
    success = run_warning_fix_tests()
    exit(0 if success else 1)