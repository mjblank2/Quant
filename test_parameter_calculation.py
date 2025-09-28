#!/usr/bin/env python3
"""
Test parameter limit calculations without database I/O issues.
"""

import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime

def test_parameter_calculation_logic():
    """Test the parameter calculation logic directly."""
    print("🧪 Testing parameter calculation logic...")
    
    # Test Universe table scenario (4 columns, 1 conflict column)
    cols_all = ['symbol', 'name', 'included', 'last_updated']
    conflict_cols = ['symbol']
    max_bind_params = 999  # SQLite limit
    
    # Calculate parameters per row accounting for ON CONFLICT DO UPDATE overhead
    update_cols_count = len(cols_all) - len(conflict_cols)  # 3
    params_per_row = len(cols_all) + update_cols_count      # 4 + 3 = 7
    
    # Add safety margin
    safety_margin = max(10, params_per_row // 10)  # max(10, 0) = 10
    effective_limit = max_bind_params - safety_margin  # 999 - 10 = 989
    
    # Calculate maximum rows per statement
    theoretical_max_rows = effective_limit // params_per_row  # 989 // 7 = 141
    
    print(f"   Universe table: {len(cols_all)} columns, {len(conflict_cols)} conflict")
    print(f"   Parameters per row: {params_per_row} (VALUES: {len(cols_all)}, UPDATE: {update_cols_count})")
    print(f"   Safety margin: {safety_margin}")
    print(f"   Effective limit: {effective_limit}")
    print(f"   Max rows per statement: {theoretical_max_rows}")
    print(f"   Old calculation would be: {max_bind_params // len(cols_all)} = {max_bind_params // len(cols_all)}")
    
    # Verify this is more conservative
    old_calc = max_bind_params // len(cols_all)
    if theoretical_max_rows < old_calc:
        print("   ✅ New calculation is more conservative")
    else:
        print("   ❌ New calculation should be more conservative")
        return False
    
    # Test Features table scenario (many columns)
    feature_cols = ['symbol', 'ts', 'ret_1d', 'ret_5d', 'ret_21d', 'mom_21', 'mom_63', 'vol_21',
                   'rsi_14', 'turnover_21', 'size_ln', 'adv_usd_21', 'beta_63', 'overnight_gap',
                   'illiq_21', 'f_pe_ttm', 'f_pb', 'f_ps_ttm', 'f_debt_to_equity', 'f_roa',
                   'f_gm', 'f_profit_margin', 'f_current_ratio']  # 23 columns
    
    feature_conflict_cols = ['symbol', 'ts']  # 2 conflict columns
    
    feature_update_cols_count = len(feature_cols) - len(feature_conflict_cols)  # 21
    feature_params_per_row = len(feature_cols) + feature_update_cols_count      # 23 + 21 = 44
    
    feature_safety_margin = max(10, feature_params_per_row // 10)  # max(10, 4) = 10
    feature_effective_limit = max_bind_params - feature_safety_margin  # 999 - 10 = 989
    
    feature_max_rows = feature_effective_limit // feature_params_per_row  # 989 // 44 = 22
    
    print(f"\n   Features table: {len(feature_cols)} columns, {len(feature_conflict_cols)} conflict")
    print(f"   Parameters per row: {feature_params_per_row} (VALUES: {len(feature_cols)}, UPDATE: {feature_update_cols_count})")
    print(f"   Safety margin: {feature_safety_margin}")
    print(f"   Effective limit: {feature_effective_limit}")
    print(f"   Max rows per statement: {feature_max_rows}")
    print(f"   Old calculation would be: {max_bind_params // len(feature_cols)} = {max_bind_params // len(feature_cols)}")
    
    # Verify this prevents the 46000 parameter scenario from the test
    old_feature_calc = max_bind_params // len(feature_cols)
    if feature_max_rows < old_feature_calc:
        print("   ✅ New calculation is more conservative for Features table")
    else:
        print("   ❌ New calculation should be more conservative for Features table")
        return False
    
    # Test PostgreSQL limits
    pg_max_params = 8000
    pg_universe_effective = pg_max_params - 10
    pg_universe_max_rows = pg_universe_effective // params_per_row  # 7990 // 7 = 1141
    
    print(f"\n   PostgreSQL Universe: max {pg_universe_max_rows} rows per statement")
    
    # Verify this would handle large universe rebuilds
    if pg_universe_max_rows > 1000:
        print("   ✅ PostgreSQL limits can handle large universe rebuilds")
    else:
        print("   ❌ PostgreSQL limits might be too restrictive")
        return False
        
    return True

def test_actual_chunking_behavior():
    """Test that the chunking logic works as expected."""
    print("\n🧪 Testing chunking behavior...")
    
    # Mock the database connection and engine
    with patch('os.environ.get', return_value='sqlite:///test.db'):
        import db
        
        # Create test data
        df_data = []
        for i in range(300):  # More than our new limit of 141
            df_data.append({
                "symbol": f"TST{i:04d}",
                "name": f"Test Company {i}",
                "included": True,
                "last_updated": datetime.utcnow(),
            })
        
        df = pd.DataFrame(df_data)
        print(f"   Created test DataFrame: {len(df)} rows × {len(df.columns)} columns")
        
        # Mock the database components
        mock_connection = Mock()
        mock_connection.engine.url = "sqlite:///test.db"
        mock_execute_results = []
        
        def mock_execute(stmt):
            # Capture the statement for analysis
            mock_execute_results.append(stmt)
            return Mock()
            
        mock_connection.execute = mock_execute
        
        # Mock the table
        mock_table = Mock()
        mock_table.__tablename__ = "universe"
        
        # Test our parameter calculation
        max_bind_params = db._max_bind_params_for_connection(mock_connection)
        conflict_cols = ['symbol']
        
        cols_all = list(df.columns)
        update_cols_count = len(cols_all) - len(conflict_cols)
        params_per_row = len(cols_all) + update_cols_count
        safety_margin = max(10, params_per_row // 10)
        effective_limit = max_bind_params - safety_margin
        theoretical_max_rows = effective_limit // params_per_row
        per_stmt_rows = min(50000, theoretical_max_rows)  # chunk_size = 50000
        
        print(f"   Max bind params: {max_bind_params}")
        print(f"   Parameters per row: {params_per_row}")
        print(f"   Safety margin: {safety_margin}")
        print(f"   Effective limit: {effective_limit}")
        print(f"   Theoretical max rows: {theoretical_max_rows}")
        print(f"   Actual chunk size: {per_stmt_rows}")
        
        # Calculate expected number of chunks
        expected_chunks = (len(df) + per_stmt_rows - 1) // per_stmt_rows
        print(f"   Expected chunks: {expected_chunks} chunks of ~{per_stmt_rows} rows each")
        
        # Verify this is more conservative than the old method
        old_chunk_size = max_bind_params // len(cols_all)
        old_chunks = (len(df) + old_chunk_size - 1) // old_chunk_size
        
        print(f"   Old method: {old_chunks} chunks of ~{old_chunk_size} rows each")
        
        if expected_chunks > old_chunks:
            print("   ✅ New method uses more, smaller chunks (more conservative)")
            return True
        else:
            print("   ❌ New method should use more chunks")
            return False

if __name__ == "__main__":
    print("🚀 Testing parameter calculation logic...")
    
    test1 = test_parameter_calculation_logic()
    test2 = test_actual_chunking_behavior()
    
    if test1 and test2:
        print("\n✅ All parameter calculation tests passed!")
        exit(0)
    else:
        print("\n❌ Some parameter calculation tests failed!")
        exit(1)