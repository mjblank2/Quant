#!/usr/bin/env python3
"""
Test to reproduce the specific parameter limit error from the problem statement.

The error shows parameters like %(symbol_m785)s::VARCHAR, %(name_m785)s::VARCHAR
up to very high numbers (e.g., m800+), suggesting bulk inserts with many parameters.
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

def create_large_parameter_scenario():
    """Create data that would trigger parameter limit errors."""
    print("🎯 Creating data that triggers parameter limit errors...")
    
    # The problem statement shows parameter names with high numbers (m785, m786, etc.)
    # This suggests a large bulk insert operation
    
    # Create a universe table bulk insert scenario with many rows
    # Universe table has: symbol, name, exchange, market_cap, adv_usd_20, included, last_updated
    
    data = []
    # Create enough rows that would exceed parameter limits when batched
    # With 7 columns per row, we need to exceed limits like:
    # SQLite: 999 params / 7 cols = ~142 rows max
    # PostgreSQL: 16000 params / 7 cols = ~2285 rows max
    
    # Create 3000 rows to test parameter limiting
    for i in range(3000):
        record = {
            'symbol': f'SYMB{i:04d}',
            'name': f'Company Name {i}',  
            'exchange': 'NASDAQ' if i % 2 == 0 else 'NYSE',
            'market_cap': 1000000.0 + i * 1000,
            'adv_usd_20': 50000.0 + i * 100,
            'included': True,
            'last_updated': datetime.utcnow()
        }
        data.append(record)
    
    return pd.DataFrame(data)

def test_parameter_limit_with_universe():
    """Test parameter limits with Universe table upserts"""
    
    # Set up test database
    db_file = tempfile.mktemp(suffix='.db')
    test_db_url = f"sqlite:///{db_file}"
    
    print(f"📊 Testing parameter limits with {test_db_url}")
    
    try:
        # Set environment for db.py
        os.environ['DATABASE_URL'] = test_db_url
        
        # Import after setting DATABASE_URL
        import db
        
        # Create tables
        db.create_tables()
        
        # Create large dataset
        large_df = create_large_parameter_scenario()
        print(f"Created DataFrame with {len(large_df)} rows and {len(large_df.columns)} columns")
        print(f"Total parameters for bulk insert: {len(large_df) * len(large_df.columns)}")
        
        # Test with different chunk sizes to trigger parameter limit issues
        conflict_cols = ['symbol']
        
        # First, test with a very large chunk size that should trigger parameter limits
        print("\n🔄 Testing with large chunk size (should trigger parameter limiting)...")
        
        try:
            db.upsert_dataframe(large_df, db.Universe, conflict_cols, chunk_size=50000)
            print("✅ Large chunk upsert completed successfully")
        except Exception as e:
            print(f"❌ Large chunk upsert failed: {e}")
            return False
            
        print("\n🔄 Testing with small chunk size (should work fine)...")
        try:
            db.upsert_dataframe(large_df, db.Universe, conflict_cols, chunk_size=100) 
            print("✅ Small chunk upsert completed successfully")
        except Exception as e:
            print(f"❌ Small chunk upsert failed: {e}")
            return False
            
        # Test parameter limit calculation
        with db.engine.connect() as conn:
            max_params = db._max_bind_params_for_connection(conn)
            print(f"📊 Max parameters for connection: {max_params}")
            
            # Calculate theoretical max rows per statement
            cols_count = len(large_df.columns)
            theoretical_max_rows = max_params // cols_count
            print(f"📊 Theoretical max rows per statement: {theoretical_max_rows} (with {cols_count} columns)")
            
        print("✅ Parameter limit test completed successfully")
        return True
        
    finally:
        # Clean up
        try:
            if os.path.exists(db_file):
                os.unlink(db_file)
        except:
            pass

def test_features_table_scenario():
    """Test with Features table which has many columns and could trigger parameter limits"""
    
    # Set up test database  
    db_file = tempfile.mktemp(suffix='.db')
    test_db_url = f"sqlite:///{db_file}"
    
    print(f"📊 Testing Features table parameter limits with {test_db_url}")
    
    try:
        # Set environment for db.py
        os.environ['DATABASE_URL'] = test_db_url
        
        # Import after setting DATABASE_URL
        import db
        
        # Create tables
        db.create_tables()
        
        # Create features data with many columns (like what would come from feature engineering)
        feature_columns = [
            'symbol', 'ts', 'ret_1d', 'ret_5d', 'ret_21d', 'mom_21', 'mom_63', 'vol_21',
            'rsi_14', 'turnover_21', 'size_ln', 'adv_usd_21', 'f_pe_ttm', 'f_pb', 'f_ps_ttm',
            'f_debt_to_equity', 'f_roa', 'f_gm', 'f_profit_margin', 'f_current_ratio',
            'beta_63', 'overnight_gap', 'illiq_21'
        ]
        
        # Create large feature dataset
        data = []
        for i in range(2000):  # Large number of records
            record = {
                'symbol': f'SYMB{i:04d}',
                'ts': date(2016, 1, 4),
                'ret_1d': np.random.normal(0, 0.02),
                'ret_5d': np.random.normal(0, 0.05),
                'ret_21d': np.random.normal(0, 0.1),
                'mom_21': np.random.normal(0, 0.15),
                'mom_63': np.random.normal(0, 0.2),
                'vol_21': np.random.lognormal(-2, 0.5),
                'rsi_14': np.random.uniform(20, 80),
                'turnover_21': np.random.lognormal(-3, 0.5),
                'size_ln': np.random.normal(10, 2),
                'adv_usd_21': np.random.lognormal(15, 1),
                'f_pe_ttm': np.random.lognormal(2.5, 0.5),
                'f_pb': np.random.lognormal(0.5, 0.5),
                'f_ps_ttm': np.random.lognormal(1, 0.5),
                'f_debt_to_equity': np.random.lognormal(-1, 0.5),
                'f_roa': np.random.normal(0.05, 0.1),
                'f_gm': np.random.uniform(0, 1),
                'f_profit_margin': np.random.normal(0.1, 0.2),
                'f_current_ratio': np.random.lognormal(0.5, 0.3),
                'beta_63': np.random.normal(1, 0.5),
                'overnight_gap': np.random.normal(0, 0.01),
                'illiq_21': np.random.lognormal(-10, 2)
            }
            data.append(record)
        
        features_df = pd.DataFrame(data)
        print(f"Created Features DataFrame with {len(features_df)} rows and {len(features_df.columns)} columns")
        print(f"Total parameters for bulk insert: {len(features_df) * len(features_df.columns)}")
        
        conflict_cols = ['symbol', 'ts']
        
        # Test with large chunk that should trigger parameter limiting 
        print("\n🔄 Testing Features table with parameter limit logic...")
        
        try:
            db.upsert_dataframe(features_df, db.Feature, conflict_cols, chunk_size=50000)
            print("✅ Features bulk upsert completed successfully")
        except Exception as e:
            print(f"❌ Features bulk upsert failed: {e}")
            # Print detailed error info
            import traceback
            traceback.print_exc()
            return False
            
        print("✅ Features table parameter limit test completed successfully")
        return True
        
    finally:
        # Clean up
        try:
            if os.path.exists(db_file):
                os.unlink(db_file)
        except:
            pass

if __name__ == "__main__":
    print("🚀 Starting parameter limit reproduction tests...")
    
    success1 = test_parameter_limit_with_universe()
    success2 = test_features_table_scenario()
    
    if success1 and success2:
        print("\n✅ All parameter limit tests passed")
    else:
        print("\n❌ Some parameter limit tests failed")
        exit(1)