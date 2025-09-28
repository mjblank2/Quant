#!/usr/bin/env python3
"""
Integration test simulating the cronjob scenario that was failing.

This test specifically validates that large universe rebuilds with ON CONFLICT
operations now work within parameter limits.
"""

import os
import tempfile
import pandas as pd
import logging
from datetime import datetime
from unittest.mock import patch

# Setup logging to match production
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def simulate_cronjob_universe_rebuild():
    """Simulate the exact cronjob scenario that was failing."""
    print("🔄 Simulating cronjob universe rebuild scenario...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Set up environment like a cronjob would
        os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
        
        # Import after setting DATABASE_URL (like in the real cron)
        import db
        from data.universe import rebuild_universe
        
        # Create database tables (like alembic would do in the cron preamble)
        db.Base.metadata.create_all(db.engine)
        
        # Simulate a large universe response from Polygon API
        # This represents the scenario where we have many small-cap stocks
        large_universe_data = []
        
        # Create enough data to potentially trigger the parameter limit error
        # The error showed m804, m805, etc., suggesting we got to around 800+ symbols
        for i in range(1000):  # 1000 symbols should be more than enough to trigger the old issue
            large_universe_data.append({
                'symbol': f'SYMB{i:04d}',
                'name': f'Small Cap Company {i} Inc.'
            })
        
        print(f"   Created mock universe data: {len(large_universe_data)} symbols")
        
        # Mock the Polygon API call to return our large dataset
        with patch('data.universe._list_small_cap_symbols', return_value=large_universe_data):
            print("   Starting universe rebuild (this was failing before the fix)...")
            
            # This is exactly what the cronjob does: python -m data.universe
            result = rebuild_universe()
            
            print(f"   ✅ Universe rebuild completed successfully!")
            print(f"   📊 Processed {len(result)} symbols")
            
            # Verify the data was actually inserted into the database
            with db.engine.connect() as conn:
                count = conn.execute(db.text("SELECT COUNT(*) FROM universe")).scalar()
                print(f"   📊 Database contains {count} universe records")
                
                # Verify we have the expected data
                sample = conn.execute(db.text("SELECT symbol, name FROM universe LIMIT 5")).fetchall()
                print("   📊 Sample records:")
                for row in sample:
                    print(f"      {row[0]}: {row[1]}")
                
        return count == len(large_universe_data)
        
    except Exception as e:
        print(f"   ❌ Cronjob simulation failed: {e}")
        # Check if it's still a parameter limit error
        error_str = str(e).lower()
        if any(phrase in error_str for phrase in ['too many variables', 'parameter limit', 'bind parameter']):
            print("   🚨 This is still a parameter limit error - fix didn't work!")
        return False
        
    finally:
        try:
            os.unlink(db_path)
        except:
            pass

def test_postgresql_scenario():
    """Test the scenario with PostgreSQL-like parameters."""
    print("\n🔄 Testing PostgreSQL scenario...")
    
    # Test our PostgreSQL parameter limits
    print("   PostgreSQL parameter limits:")
    
    # Mock PostgreSQL connection
    class MockPGConnection:
        class MockEngine:
            class MockUrl:
                def __lower__(self):
                    return "postgresql+psycopg://user:pass@host/db"
                def __str__(self):
                    return "postgresql+psycopg://user:pass@host/db"
            url = MockUrl()
        engine = MockEngine()
    
    import db
    pg_limit = db._max_bind_params_for_connection(MockPGConnection())
    print(f"   Max bind params for PostgreSQL: {pg_limit}")
    
    # Calculate universe table limits for PostgreSQL  
    universe_cols = 4  # symbol, name, included, last_updated
    conflict_cols = 1  # symbol
    update_cols = universe_cols - conflict_cols  # 3
    params_per_row = universe_cols + update_cols  # 7
    safety_margin = max(10, params_per_row // 10)  # 10
    effective_limit = pg_limit - safety_margin
    max_rows_per_stmt = effective_limit // params_per_row
    
    print(f"   Parameters per row: {params_per_row}")
    print(f"   Safety margin: {safety_margin}")
    print(f"   Effective limit: {effective_limit}")
    print(f"   Max rows per statement: {max_rows_per_stmt}")
    
    # This should be able to handle reasonable universe sizes
    if max_rows_per_stmt >= 1000:
        print("   ✅ PostgreSQL can handle large universe rebuilds")
        return True
    else:
        print("   ❌ PostgreSQL limits might be too restrictive")
        return False

def test_features_scenario():
    """Test the Features table scenario that could also hit parameter limits."""
    print("\n🔄 Testing Features table scenario...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
        
        import db
        
        # Create database tables
        db.Base.metadata.create_all(db.engine)
        
        # Create a realistic features dataset
        # Features table has many columns, so this could easily hit parameter limits
        feature_data = []
        base_date = pd.to_datetime('2024-01-01').date()
        
        # Simulate daily feature calculation for multiple symbols
        symbols = [f'FEAT{i:03d}' for i in range(100)]  # 100 symbols
        
        for symbol in symbols:
            feature_data.append({
                'symbol': symbol,
                'ts': base_date,
                'ret_1d': 0.01,
                'ret_5d': 0.05, 
                'ret_21d': 0.10,
                'vol_21': 0.20,
                'size_ln': 15.0,
                'adv_usd_21': 1000000.0,
                'mom_21': 0.15,
                'mom_63': 0.25,
                'rsi_14': 50.0,
                'turnover_21': 0.05,
                'beta_63': 1.0,
                'overnight_gap': 0.001,
                'illiq_21': 0.0001,
                'f_pe_ttm': 15.0,
                'f_pb': 2.0,
                'f_ps_ttm': 3.0,
                'f_debt_to_equity': 0.5,
                'f_roa': 0.1,
                'f_gm': 0.3,
                'f_profit_margin': 0.1,
                'f_current_ratio': 2.0
            })
            
        df = pd.DataFrame(feature_data)
        print(f"   Created Features DataFrame: {len(df)} rows × {len(df.columns)} columns")
        
        # Calculate expected parameter usage
        conflict_cols = ['symbol', 'ts']
        total_params = len(df) * (len(df.columns) + (len(df.columns) - len(conflict_cols)))
        print(f"   Total parameters needed: {total_params}")
        
        # This should work with the new parameter limiting
        try:
            db.upsert_dataframe(df, db.Feature, conflict_cols=conflict_cols)
            print("   ✅ Features upsert completed successfully")
            
            # Verify insertion
            with db.engine.connect() as conn:
                count = conn.execute(db.text("SELECT COUNT(*) FROM features")).scalar()
                print(f"   📊 Database contains {count} feature records")
                
            return True
            
        except Exception as e:
            print(f"   ❌ Features upsert failed: {e}")
            return False
            
    finally:
        try:
            os.unlink(db_path)
        except:
            pass

if __name__ == "__main__":
    print("🚀 Running cronjob scenario integration test...")
    print("=" * 60)
    
    test1 = simulate_cronjob_universe_rebuild()
    test2 = test_postgresql_scenario() 
    test3 = test_features_scenario()
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST RESULTS:")
    print(f"✅ Cronjob universe rebuild: {'PASS' if test1 else 'FAIL'}")
    print(f"✅ PostgreSQL parameter limits: {'PASS' if test2 else 'FAIL'}")
    print(f"✅ Features table scenario: {'PASS' if test3 else 'FAIL'}")
    
    if test1 and test2 and test3:
        print("\n🎉 All integration tests passed!")
        print("🔧 The parameter limit issue should now be resolved!")
        exit(0)
    else:
        print("\n❌ Some integration tests failed!")
        exit(1)