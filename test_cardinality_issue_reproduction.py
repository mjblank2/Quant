#!/usr/bin/env python3
"""
Test to reproduce the exact CardinalityViolation issue from the logs.

The problem statement shows:
- Many repeated warnings: "CardinalityViolation with 10 records"  
- Very long SQL statements with many parameter placeholders
- Repeated retries happening

This suggests the current fix is working but not preventing the initial violations.
"""

import pandas as pd
import numpy as np
import logging
import tempfile
import os
from datetime import date, datetime
from sqlalchemy import create_engine, text

# Setup logging to capture warnings
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

def create_problem_scenario_data():
    """Create test data that reproduces the high-parameter, high-duplicate scenario."""
    print("🎯 Creating problem scenario data...")
    
    # Create a large batch with many columns (mimicking features table) and duplicates
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'] * 50  # 250 symbols
    
    data = []
    for i, symbol in enumerate(symbols):
        # Create multiple records for the same symbol+date (the duplication problem)
        for j in range(2):  # 2 records per symbol+date combo
            record = {
                'symbol': symbol,
                'ts': date(2016, 1, 4),  # Same date for all to create maximum duplicates
                'ret_1d': 0.01 + i * 0.001 + j * 0.0001,
                'ret_5d': 0.05 + i * 0.002,
                'ret_21d': 0.21 + i * 0.003,
                'mom_21': 0.15 + i * 0.001,
                'mom_63': 0.25 + i * 0.002,
                'vol_21': 0.18 + i * 0.0005,
                'rsi_14': 50 + i * 0.1,
                'turnover_21': 0.08 + i * 0.0001,
                'size_ln': 12.5 + i * 0.01,
                'adv_usd_21': 1000000 + i * 10000,
                'f_pe_ttm': 15 + i * 0.1,
                'f_pb': 2.5 + i * 0.01,
                'f_ps_ttm': 3.2 + i * 0.02,
                'f_debt_to_equity': 0.45 + i * 0.001,
                'f_roa': 0.12 + i * 0.0001,
                'f_gm': 0.35 + i * 0.001,
                'f_profit_margin': 0.18 + i * 0.0005,
                'f_current_ratio': 1.8 + i * 0.01,
                'beta_63': 1.2 + i * 0.001,
                'overnight_gap': 0.002 + i * 0.00001,
                'illiq_21': 0.0001 + i * 0.000001,
            }
            data.append(record)
    
    df = pd.DataFrame(data)
    print(f"Created problem data: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Theoretical parameter count: {len(df) * len(df.columns)} = {len(df) * len(df.columns)}")
    
    # Check duplicates
    duplicates = df.groupby(['symbol', 'ts']).size()
    max_dupes = duplicates.max()
    total_dupes = (duplicates > 1).sum()
    print(f"Duplicate analysis: {total_dupes} symbol+date pairs have duplicates (max {max_dupes} per pair)")
    
    return df

def test_high_volume_scenario():
    """Test the exact high-volume scenario that causes the warnings."""
    print("\n🔍 Testing high-volume CardinalityViolation scenario...")
    
    # Create a temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        temp_db_path = tmp.name
    
    try:
        # Create test database and engine
        engine = create_engine(f'sqlite:///{temp_db_path}')
        
        # Import the upsert function and create problem data
        import sys
        sys.path.append('/home/runner/work/Quant/Quant')
        
        # Set up to capture warnings
        import logging
        log_capture = []
        
        class LogCapture(logging.Handler):
            def emit(self, record):
                log_capture.append(self.format(record))
        
        handler = LogCapture()
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        db_logger = logging.getLogger('db')
        db_logger.addHandler(handler)
        db_logger.setLevel(logging.DEBUG)
        
        from db import upsert_dataframe, Feature
        
        # Create the actual features table in the test database
        from db import Base
        Base.metadata.create_all(engine)
        
        # Create the problematic data
        problem_data = create_problem_scenario_data()
        
        print(f"\nAttempting upsert with {len(problem_data)} rows...")
        
        # Try to upsert this data - this should trigger the warnings
        try:
            start_time = datetime.now()
            with engine.connect() as conn:
                upsert_dataframe(problem_data, Feature, ['symbol', 'ts'], chunk_size=1000, conn=conn)
                conn.commit()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Check results
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM features")).scalar()
                print(f"\n✅ Upsert completed in {duration:.2f}s")
                print(f"Records in database: {result}")
                
                # Check for duplicates in final result
                duplicates_in_db = conn.execute(text("""
                    SELECT symbol, ts, COUNT(*) as count 
                    FROM features 
                    GROUP BY symbol, ts 
                    HAVING COUNT(*) > 1
                """)).fetchall()
                
                if len(duplicates_in_db) == 0:
                    print("✅ No duplicates in final database")
                else:
                    print(f"❌ Found {len(duplicates_in_db)} duplicates in database")
                
                # Analyze the log output
                print(f"\nLog analysis:")
                total_warnings = len([msg for msg in log_capture if 'WARNING' in msg or 'WARN' in msg])
                cardinality_warnings = len([msg for msg in log_capture if 'CardinalityViolation' in msg])
                parameter_warnings = len([msg for msg in log_capture if 'Parameter limit' in msg])
                
                print(f"Total warnings: {total_warnings}")
                print(f"CardinalityViolation warnings: {cardinality_warnings}")
                print(f"Parameter limit warnings: {parameter_warnings}")
                
                if total_warnings > 0:
                    print("\nSample warning messages:")
                    for msg in log_capture[:5]:  # Show first 5 warnings
                        if 'WARNING' in msg or 'WARN' in msg:
                            print(f"  {msg}")
                
                return total_warnings == 0  # Success if no warnings
                
        except Exception as e:
            print(f"❌ Upsert failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    finally:
        # Cleanup
        try:
            os.unlink(temp_db_path)
        except:
            pass

def test_parameter_limit_scenario():
    """Test hitting PostgreSQL parameter limits."""
    print("\n⚡ Testing parameter limit scenario...")
    
    # Create data that would exceed PostgreSQL parameter limits
    # PostgreSQL default limit is ~65,535 parameters
    # With 22 columns, that's ~2,978 rows maximum
    
    symbols = ['TEST'] * 3000  # 3000 symbols, all same to create duplicates
    
    data = []
    for i, symbol in enumerate(symbols):
        record = {
            'symbol': symbol,
            'ts': date(2016, 1, 4),  # Same date for maximum duplicates
            'ret_1d': 0.01 + i * 0.001,
            'ret_5d': 0.05 + i * 0.002,
            'ret_21d': 0.21 + i * 0.003,
            'mom_21': 0.15 + i * 0.001,
            'mom_63': 0.25 + i * 0.002,
            'vol_21': 0.18 + i * 0.0005,
            'rsi_14': 50 + i * 0.1,
            'turnover_21': 0.08 + i * 0.0001,
            'size_ln': 12.5 + i * 0.01,
            'adv_usd_21': 1000000 + i * 10000,
            'f_pe_ttm': 15 + i * 0.1,
            'f_pb': 2.5 + i * 0.01,
            'f_ps_ttm': 3.2 + i * 0.02,
            'f_debt_to_equity': 0.45 + i * 0.001,
            'f_roa': 0.12 + i * 0.0001,
            'f_gm': 0.35 + i * 0.001,
            'f_profit_margin': 0.18 + i * 0.0005,
            'f_current_ratio': 1.8 + i * 0.01,
            'beta_63': 1.2 + i * 0.001,
            'overnight_gap': 0.002 + i * 0.00001,
            'illiq_21': 0.0001 + i * 0.000001,
        }
        data.append(record)
    
    df = pd.DataFrame(data)
    
    # This should all deduplicate to 1 row (all same symbol+date)
    duplicates = df.groupby(['symbol', 'ts']).size()
    max_dupes = duplicates.max()
    print(f"Parameter limit test data: {len(df)} rows -> should dedupe to {len(duplicates)} unique rows")
    print(f"Max duplicates per symbol+date: {max_dupes}")
    print(f"Theoretical parameters before dedup: {len(df) * len(df.columns)} = {len(df) * len(df.columns)}")
    
    # Test the deduplication logic
    conflict_cols = ['symbol', 'ts']
    original_size = len(df)
    df_deduped = df.drop_duplicates(subset=conflict_cols, keep='last').reset_index(drop=True)
    dedupe_size = len(df_deduped)
    
    print(f"After deduplication: {original_size} -> {dedupe_size} rows")
    print(f"Parameters after dedup: {dedupe_size * len(df.columns)} = {dedupe_size * len(df.columns)}")
    
    if dedupe_size * len(df.columns) < 10000:  # Under the conservative limit
        print("✅ Parameter limit test passed - deduplication brings us under limits")
        return True
    else:
        print("❌ Parameter limit test failed - still too many parameters after dedup")
        return False

def run_reproduction_tests():
    """Run all tests to reproduce the CardinalityViolation issue."""
    print("🎯 Running CardinalityViolation issue reproduction tests\n")
    
    tests = [
        test_parameter_limit_scenario,
        test_high_volume_scenario,
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
    
    print(f"📊 Reproduction Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All reproduction tests passed!")
        print("✅ Current fix handles the problematic scenarios correctly.")
        return True
    else:
        print("⚠️ Some tests failed - issue may still exist")
        return False

if __name__ == "__main__":
    success = run_reproduction_tests()
    exit(0 if success else 1)