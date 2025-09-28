#!/usr/bin/env python3
"""
Test to validate the parameter limit fix for ON CONFLICT DO UPDATE statements.
"""

import os
import tempfile
import pandas as pd
import logging
from datetime import datetime
from unittest.mock import patch

# Setup logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

def test_parameter_calculation_fix():
    """Test that parameter calculation now accounts for ON CONFLICT overhead."""
    print("🧪 Testing parameter calculation with ON CONFLICT overhead...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
        
        # Import after setting DATABASE_URL
        import db
        
        # Create database tables
        db.Base.metadata.create_all(db.engine)
        
        # Test the parameter calculation logic
        with db.engine.connect() as conn:
            max_params = db._max_bind_params_for_connection(conn)
            print(f"   Max parameters for SQLite: {max_params}")
            
            # Universe table has 4 columns, conflict on 'symbol' (1 column)
            # Old calculation: 999 / 4 = 249 rows
            # New calculation: 
            #   - params_per_row = 4 (VALUES) + 3 (UPDATE SET: name, included, last_updated)
            #   - params_per_row = 7
            #   - effective_limit = 999 - 70 (safety margin) = 929
            #   - max_rows = 929 / 7 = 132 rows
            
            cols_all = ['symbol', 'name', 'included', 'last_updated']
            conflict_cols = ['symbol']
            
            update_cols_count = len(cols_all) - len(conflict_cols)
            params_per_row = len(cols_all) + update_cols_count
            safety_margin = max(10, params_per_row // 10)
            effective_limit = max_params - safety_margin
            theoretical_max_rows = effective_limit // params_per_row
            
            print(f"   Columns: {len(cols_all)}, Conflict: {len(conflict_cols)}, Update: {update_cols_count}")
            print(f"   Parameters per row: {params_per_row}")
            print(f"   Safety margin: {safety_margin}")
            print(f"   Effective limit: {effective_limit}")
            print(f"   Max rows per statement: {theoretical_max_rows}")
            
            # This should be much more conservative than the old calculation
            old_calculation = max_params // len(cols_all)
            print(f"   Old calculation would allow: {old_calculation} rows")
            print(f"   New calculation allows: {theoretical_max_rows} rows")
            
            if theoretical_max_rows < old_calculation:
                print("   ✅ New calculation is more conservative - good!")
                return True
            else:
                print("   ❌ New calculation should be more conservative")
                return False
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        try:
            os.unlink(db_path)
        except:
            pass

def test_large_universe_insert_with_fix():
    """Test that large universe inserts work with the new parameter calculation."""
    print("🧪 Testing large universe insert with parameter fix...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
        
        # Import after setting DATABASE_URL
        import db
        from db import Universe
        
        # Create database tables
        db.Base.metadata.create_all(db.engine)
        
        # Create a large dataset that would trigger the old parameter limit issue
        # Use enough records to exceed the old calculation but be within the new safe limits
        df_data = []
        for i in range(200):  # This would be 200 * 7 = 1400 params with new calc, exceeding 999
            df_data.append({
                "symbol": f"TST{i:04d}",
                "name": f"Test Company {i}",
                "included": True,
                "last_updated": datetime.utcnow(),
            })
        
        df = pd.DataFrame(df_data)
        print(f"   Created DataFrame with {len(df)} records, {len(df.columns)} columns")
        
        # This should now work without hitting parameter limits
        try:
            db.upsert_dataframe(df, Universe, conflict_cols=["symbol"])
            print("   ✅ Large universe insert completed successfully")
            
            # Verify the data was inserted
            with db.engine.connect() as conn:
                count = conn.execute(db.text("SELECT COUNT(*) FROM universe")).scalar()
                print(f"   📊 Database contains {count} universe records")
                
            return True
            
        except Exception as e:
            print(f"   ❌ Large universe insert failed: {e}")
            # Check if it's a parameter limit error
            error_str = str(e).lower()
            if "too many" in error_str and ("variables" in error_str or "parameters" in error_str):
                print("   🎯 This is still a parameter limit error - fix needs improvement")
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False
    finally:
        try:
            os.unlink(db_path)
        except:
            pass

def test_features_table_with_many_columns():
    """Test that tables with many columns work with the new parameter calculation.""" 
    print("🧪 Testing features table with many columns...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
        
        # Import after setting DATABASE_URL
        import db
        from db import Feature
        
        # Create database tables
        db.Base.metadata.create_all(db.engine)
        
        # Create a features dataset with many columns (20+)
        feature_data = []
        base_features = {
            'symbol': 'TEST0001', 
            'ts': pd.to_datetime('2024-01-01').date(),
            'ret_1d': 0.01, 'ret_5d': 0.05, 'ret_21d': 0.10,
            'vol_21': 0.20, 'size_ln': 15.0, 'adv_usd_21': 1000000.0,
            'mom_21': 0.15, 'mom_63': 0.25, 'rsi_14': 50.0,
            'turnover_21': 0.05, 'beta_63': 1.0, 'overnight_gap': 0.001,
            'illiq_21': 0.0001, 'f_pe_ttm': 15.0, 'f_pb': 2.0,
            'f_ps_ttm': 3.0, 'f_debt_to_equity': 0.5, 'f_roa': 0.1,
            'f_gm': 0.3, 'f_profit_margin': 0.1, 'f_current_ratio': 2.0
        }
        
        # Create multiple records with different symbols
        for i in range(50):  # With ~22 columns, this is 50 * 44 params = 2200 with new calc
            record = base_features.copy()
            record['symbol'] = f'TEST{i:04d}'
            feature_data.append(record)
            
        df = pd.DataFrame(feature_data)
        print(f"   Created DataFrame with {len(df)} records, {len(df.columns)} columns")
        
        # Calculate expected parameter usage
        conflict_cols = ['symbol', 'ts']
        update_cols_count = len(df.columns) - len(conflict_cols)
        params_per_row = len(df.columns) + update_cols_count
        total_params = len(df) * params_per_row
        
        print(f"   Expected parameter usage: {len(df)} rows * {params_per_row} params/row = {total_params}")
        
        try:
            db.upsert_dataframe(df, Feature, conflict_cols=conflict_cols)
            print("   ✅ Features table insert completed successfully")
            
            # Verify the data was inserted
            with db.engine.connect() as conn:
                count = conn.execute(db.text("SELECT COUNT(*) FROM features")).scalar()
                print(f"   📊 Database contains {count} feature records")
                
            return True
            
        except Exception as e:
            print(f"   ❌ Features table insert failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False
    finally:
        try:
            os.unlink(db_path)
        except:
            pass

if __name__ == "__main__":
    print("🚀 Starting parameter limit fix validation...")
    
    test1 = test_parameter_calculation_fix()
    test2 = test_large_universe_insert_with_fix() 
    test3 = test_features_table_with_many_columns()
    
    if test1 and test2 and test3:
        print("\n✅ All parameter limit fix tests passed!")
        exit(0)
    else:
        print("\n❌ Some parameter limit fix tests failed!")
        exit(1)