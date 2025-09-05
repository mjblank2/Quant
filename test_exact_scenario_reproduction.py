#!/usr/bin/env python3
"""
Test to reproduce the exact scenario from the problem statement logs.

The logs show:
- "CardinalityViolation with 10 records, deduping on conflict cols ['symbol', 'ts'] and retrying in smaller batches"
- Multiple occurrences of this message
- Parameter limit warnings

This suggests the issue might be happening during the retry phase, not the initial phase.
"""

import pandas as pd
import numpy as np
import logging
import tempfile
import os
from datetime import date, datetime
from sqlalchemy import create_engine, text

# Setup detailed logging to capture all levels
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')
log = logging.getLogger(__name__)

def create_retry_scenario_data():
    """Create data that would cause retries during upsert."""
    print("🎯 Creating data that triggers retry scenarios...")
    
    # Create data that would cause the retry mechanism to kick in
    # The logs show "10 records" in the retry, so let's create batches that would 
    # result in 10-record retries
    
    data = []
    # Create 50 records that will all be duplicates after initial processing
    for i in range(50):
        record = {
            'symbol': 'AAPL',  # All same symbol
            'ts': date(2016, 1, 4),  # All same date
            'ret_1d': 0.01 + i * 0.001,  # Different values to avoid exact duplicates initially
            'ret_5d': 0.05,
            'ret_21d': 0.21,
            'mom_21': 0.15,
            'mom_63': 0.25,
            'vol_21': 0.18,
            'rsi_14': 50.0,
            'turnover_21': 0.08,
            'size_ln': 12.5,
            'adv_usd_21': 1000000.0,
            'f_pe_ttm': 15.0,
            'f_pb': 2.5,
            'f_ps_ttm': 3.2,
            'f_debt_to_equity': 0.45,
            'f_roa': 0.12,
            'f_gm': 0.35,
            'f_profit_margin': 0.18,
            'f_current_ratio': 1.8,
            'beta_63': 1.2,
            'overnight_gap': 0.002,
            'illiq_21': 0.0001,
        }
        data.append(record)
    
    df = pd.DataFrame(data)
    print(f"Created retry scenario data: {len(df)} rows")
    
    # Check what happens when we process this in chunks
    chunk_size = 10  # This matches the "10 records" from the logs
    
    for i, start in enumerate(range(0, len(df), chunk_size)):
        chunk = df.iloc[start:start + chunk_size]
        duplicates = chunk.groupby(['symbol', 'ts']).size()
        max_dupes = duplicates.max() if len(duplicates) > 0 else 0
        print(f"Chunk {i+1}: {len(chunk)} rows, max duplicates per symbol+date: {max_dupes}")
    
    return df

def simulate_failing_batch_insert():
    """Simulate what happens when a batch insert fails and triggers retry."""
    print("\n🔄 Simulating failing batch insert scenario...")
    
    # This simulates the exact scenario where the retry mechanism would kick in
    # We'll manually trigger the conditions that cause the warnings
    
    import sys
    sys.path.append('/home/runner/work/Quant/Quant')
    
    # Create data that would cause duplicate issues during batch processing
    records = []
    for i in range(10):  # 10 records as mentioned in the logs
        record = {
            'symbol': 'AAPL',
            'ts': date(2016, 1, 4),
            'ret_1d': 0.01,  # Same values - these are true duplicates after conflict resolution
            'ret_5d': 0.05,
            'ret_21d': 0.21,
            'mom_21': 0.15,
            'mom_63': 0.25,
            'vol_21': 0.18,
            'rsi_14': 50.0,
            'turnover_21': 0.08,
            'size_ln': 12.5,
            'adv_usd_21': 1000000.0,
            'f_pe_ttm': 15.0,
            'f_pb': 2.5,
            'f_ps_ttm': 3.2,
            'f_debt_to_equity': 0.45,
            'f_roa': 0.12,
            'f_gm': 0.35,
            'f_profit_margin': 0.18,
            'f_current_ratio': 1.8,
            'beta_63': 1.2,
            'overnight_gap': 0.002,
            'illiq_21': 0.0001,
        }
        records.append(record)
    
    # Rebuild DataFrame from records (this simulates what happens in the retry logic)
    smaller_df = pd.DataFrame(records)
    print(f"Simulated retry batch: {len(smaller_df)} records")
    print(f"Sample data:")
    print(smaller_df[['symbol', 'ts', 'ret_1d']].head())
    
    # Apply the deduplication logic from db.py retry
    conflict_cols = ['symbol', 'ts']
    
    if len(smaller_df) > 0 and conflict_cols and set(conflict_cols).issubset(smaller_df.columns):
        original_size = len(smaller_df)
        smaller_df = smaller_df.drop_duplicates(subset=conflict_cols, keep='last').reset_index(drop=True)
        dedupe_size = len(smaller_df)
        if dedupe_size < original_size:
            print(f"⚠️  Retry deduplication: {original_size} -> {dedupe_size} rows")
            print(f"This would trigger: 'Removed {original_size - dedupe_size} duplicate rows during retry to prevent CardinalityViolation'")
    
    print(f"Final retry batch size: {len(smaller_df)}")
    return len(smaller_df) == 1  # Should dedupe to 1 row

def test_multiple_concurrent_batches():
    """Test what happens with multiple batches that all have the same conflict keys."""
    print("\n🔀 Testing multiple concurrent batches scenario...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        temp_db_path = tmp.name
    
    try:
        engine = create_engine(f'sqlite:///{temp_db_path}')
        
        import sys
        sys.path.append('/home/runner/work/Quant/Quant')
        
        # Set up to capture warnings
        log_capture = []
        
        class LogCapture(logging.Handler):
            def emit(self, record):
                log_capture.append(self.format(record))
        
        handler = LogCapture()
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        db_logger = logging.getLogger('db')
        db_logger.addHandler(handler)
        db_logger.setLevel(logging.DEBUG)
        
        from db import upsert_dataframe, Feature, Base
        Base.metadata.create_all(engine)
        
        # Create multiple batches that would all conflict with each other
        all_data = []
        
        # Batch 1: 10 records, all AAPL on same date
        for i in range(10):
            all_data.append({
                'symbol': 'AAPL',
                'ts': date(2016, 1, 4),
                'ret_1d': 0.01 + i * 0.001,  # Slightly different values
                'vol_21': 0.15 + i * 0.001,
            })
        
        # Batch 2: 10 more records, same AAPL on same date
        for i in range(10):
            all_data.append({
                'symbol': 'AAPL',
                'ts': date(2016, 1, 4),
                'ret_1d': 0.02 + i * 0.001,  # Different values
                'vol_21': 0.16 + i * 0.001,
            })
        
        df = pd.DataFrame(all_data)
        print(f"Testing with {len(df)} records that should all dedupe to 1 final record")
        
        # Force small chunk size to trigger multiple batches
        print("Attempting upsert with very small chunk size to force retries...")
        
        try:
            with engine.connect() as conn:
                # Use chunk_size=5 to force the data to be split into multiple batches
                upsert_dataframe(df, Feature, ['symbol', 'ts'], chunk_size=5, conn=conn)
                conn.commit()
            
            # Check results
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM features")).scalar()
                print(f"Final records in database: {result}")
                
                actual_data = conn.execute(text("SELECT symbol, ts, ret_1d, vol_21 FROM features")).fetchone()
                if actual_data:
                    print(f"Final record: {actual_data}")
            
            # Analyze log output
            print(f"\nLog analysis:")
            total_warnings = len([msg for msg in log_capture if 'WARNING' in msg])
            cardinality_warnings = len([msg for msg in log_capture if 'CardinalityViolation' in msg])
            parameter_warnings = len([msg for msg in log_capture if 'Parameter limit' in msg])
            
            print(f"Total warnings: {total_warnings}")
            print(f"CardinalityViolation warnings: {cardinality_warnings}")
            print(f"Parameter limit warnings: {parameter_warnings}")
            
            if total_warnings > 0:
                print("\nWarning messages:")
                for msg in log_capture:
                    if 'WARNING' in msg:
                        print(f"  {msg}")
            
            # Success if we ended up with 1 record and no cardinality violations
            return result == 1 and cardinality_warnings == 0
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    finally:
        try:
            os.unlink(temp_db_path)
        except:
            pass

def run_scenario_analysis():
    """Run all scenario analysis tests."""
    print("🎯 Running CardinalityViolation scenario analysis\n")
    
    tests = [
        simulate_failing_batch_insert,
        test_multiple_concurrent_batches,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print("✅ Test passed")
            else:
                print("❌ Test failed") 
            print()
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print(f"📊 Scenario Analysis Results: {passed}/{total} tests passed")
    
    if passed != total:
        print("⚠️ Some scenarios show issues that need to be addressed")
    
    return passed == total

if __name__ == "__main__":
    # First understand the data characteristics
    create_retry_scenario_data()
    
    # Then run the scenario tests
    success = run_scenario_analysis()
    exit(0 if success else 1)