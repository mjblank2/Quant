#!/usr/bin/env python3
"""
Test to reproduce the exact production scenario that's still failing.
This test creates a scenario that matches the original error message exactly.
"""

import pandas as pd
import numpy as np
import logging
import tempfile
import os
from datetime import date
from sqlalchemy import create_engine, text

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def test_production_scenario():
    """Test the exact scenario from the production error."""
    print("🎯 Testing production CardinalityViolation scenario...")
    
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
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from db import Base, Feature, upsert_dataframe
        
        # Create tables
        Base.metadata.create_all(engine)
        
        # Create data that matches the production error exactly
        # The error shows INSERT with all these fields
        fields = [
            'symbol', 'ts', 'ret_1d', 'ret_5d', 'ret_21d', 'mom_21', 'mom_63', 
            'vol_21', 'rsi_14', 'turnover_21', 'size_ln', 'adv_usd_21', 'f_pe_ttm', 
            'f_pb', 'f_ps_ttm', 'f_debt_to_equity', 'f_roa', 'f_gm', 'f_profit_margin', 
            'f_current_ratio', 'beta_63', 'overnight_gap', 'illiq_21'
        ]
        
        # Create problematic data - large batch with duplicates like would happen in production
        data = []
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX']
        dates = [date(2016, 1, 4), date(2016, 1, 5), date(2016, 1, 6)]
        
        # First, create regular data
        for symbol in symbols:
            for dt in dates:
                record = {
                    'symbol': symbol,
                    'ts': dt,
                    'ret_1d': np.random.normal(0, 0.02),
                    'ret_5d': np.random.normal(0, 0.05),
                    'ret_21d': np.random.normal(0, 0.1),
                    'mom_21': np.random.normal(0, 0.15),
                    'mom_63': np.random.normal(0, 0.25),
                    'vol_21': np.random.uniform(0.1, 0.3),
                    'rsi_14': np.random.uniform(20, 80),
                    'turnover_21': np.random.uniform(0.01, 0.2),
                    'size_ln': np.random.uniform(10, 15),
                    'adv_usd_21': np.random.uniform(100000, 10000000),
                    'f_pe_ttm': np.random.uniform(10, 30),
                    'f_pb': np.random.uniform(1, 5),
                    'f_ps_ttm': np.random.uniform(2, 8),
                    'f_debt_to_equity': np.random.uniform(0, 1),
                    'f_roa': np.random.uniform(0.05, 0.25),
                    'f_gm': np.random.uniform(0.2, 0.6),
                    'f_profit_margin': np.random.uniform(0.1, 0.3),
                    'f_current_ratio': np.random.uniform(1, 3),
                    'beta_63': np.random.uniform(0.5, 2),
                    'overnight_gap': np.random.normal(0, 0.01),
                    'illiq_21': np.random.uniform(0.0001, 0.01),
                }
                data.append(record)
        
        # Now add duplicates that would cause the CardinalityViolation
        # Add duplicate rows for the same (symbol, ts) pairs with slightly different values
        for i in range(10):  # Add 10 duplicates
            duplicate_record = data[i].copy()
            # Change some values slightly to simulate feature engineering creating duplicates
            duplicate_record['ret_1d'] = duplicate_record['ret_1d'] + 0.001
            duplicate_record['vol_21'] = duplicate_record['vol_21'] + 0.01
            data.append(duplicate_record)
        
        df = pd.DataFrame(data)
        print(f"Created test data: {len(df)} rows")
        
        # Check for duplicates
        duplicates = df.groupby(['symbol', 'ts']).size()
        max_dupes = duplicates.max()
        total_dupes = len(duplicates[duplicates > 1])
        print(f"Duplicates in test data: {total_dupes} groups with max {max_dupes} rows per group")
        
        # Test 1: Try upsert with default chunk_size (like production)
        print("\n🔧 Test 1: Default chunk_size upsert (like production)...")
        try:
            with engine.connect() as conn:
                # This should work with the proactive deduplication
                upsert_dataframe(df, Feature, ['symbol', 'ts'], conn=conn)
                conn.commit()
            
            # Verify results
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM features")).scalar()
                print(f"✅ Successfully inserted {result} records")
                
                # Check for duplicates in database
                db_duplicates = conn.execute(text("""
                    SELECT symbol, ts, COUNT(*) as count 
                    FROM features 
                    GROUP BY symbol, ts 
                    HAVING COUNT(*) > 1
                """)).fetchall()
                
                if len(db_duplicates) == 0:
                    print("✅ No duplicates in database - fix working correctly")
                    return True
                else:
                    print(f"❌ Found {len(db_duplicates)} duplicates in database")
                    return False
                    
        except Exception as e:
            print(f"❌ Test 1 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    finally:
        # Cleanup
        try:
            os.unlink(temp_db_path)
        except:
            pass

def test_large_batch_scenario():
    """Test with a very large batch that might trigger different chunking behavior."""
    print("\n🎯 Testing large batch scenario...")
    
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
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from db import Base, Feature, upsert_dataframe
        
        # Create tables
        Base.metadata.create_all(engine)
        
        # Create a large dataset like production would have
        data = []
        for i in range(1000):  # Large dataset
            symbol = f"SYM{i % 100:03d}"  # 100 different symbols
            ts = date(2016, 1, 4)  # Same date for all - this will create duplicates
            record = {
                'symbol': symbol,
                'ts': ts,
                'ret_1d': np.random.normal(0, 0.02),
                'ret_5d': np.random.normal(0, 0.05),
                'ret_21d': np.random.normal(0, 0.1),
                'mom_21': np.random.normal(0, 0.15),
                'mom_63': np.random.normal(0, 0.25),
                'vol_21': np.random.uniform(0.1, 0.3),
                'rsi_14': np.random.uniform(20, 80),
                'turnover_21': np.random.uniform(0.01, 0.2),
                'size_ln': np.random.uniform(10, 15),
                'adv_usd_21': np.random.uniform(100000, 10000000),
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        print(f"Created large test data: {len(df)} rows")
        
        # Check for duplicates
        duplicates = df.groupby(['symbol', 'ts']).size()
        max_dupes = duplicates.max()
        total_dupes = len(duplicates[duplicates > 1])
        print(f"Duplicates in large test data: {total_dupes} groups with max {max_dupes} rows per group")
        
        try:
            with engine.connect() as conn:
                upsert_dataframe(df, Feature, ['symbol', 'ts'], conn=conn)
                conn.commit()
            
            # Verify results
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM features")).scalar()
                print(f"✅ Successfully inserted {result} records from large batch")
                return True
                
        except Exception as e:
            print(f"❌ Large batch test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    finally:
        # Cleanup
        try:
            os.unlink(temp_db_path)
        except:
            pass

if __name__ == "__main__":
    print("🔬 Running production scenario reproduction tests...\n")
    
    success1 = test_production_scenario()
    success2 = test_large_batch_scenario()
    
    if success1 and success2:
        print("\n🎉 All production scenario tests passed!")
        print("✅ The fix should prevent CardinalityViolation errors in production.")
    else:
        print("\n❌ Some production scenario tests failed!")
        print("🔧 The fix may need additional improvements.")
    
    exit(0 if (success1 and success2) else 1)