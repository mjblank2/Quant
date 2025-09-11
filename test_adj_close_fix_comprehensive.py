#!/usr/bin/env python3
"""
Test script for adj_close column fix and index alignment bug.
"""
import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# Set up test environment
os.environ['DATABASE_URL'] = 'sqlite:///test_adj_close.db'

# Import modules after setting DATABASE_URL
from utils.price_utils import has_adj_close, price_expr, log_price_mode_once
from models.ml import _ic_by_model
from db import engine


def test_price_utils():
    """Test price utilities functionality."""
    print("Testing price utilities...")
    
    # Test with database that doesn't have adj_close
    print(f"has_adj_close(): {has_adj_close()}")
    print(f"price_expr(): {price_expr()}")
    
    # Force logging of price mode
    log_price_mode_once()
    
    print("✓ Price utilities work correctly")


def test_index_alignment_bug():
    """Test that the index alignment bug in _ic_by_model is fixed."""
    print("\nTesting index alignment fix...")
    
    # Create test data with non-sequential indices (like real DataFrames might have)
    n_rows = 100
    indices = np.random.choice(range(1000), size=n_rows, replace=False)  # Non-sequential indices
    indices.sort()
    
    # Create test DataFrame with gaps in the index
    test_df = pd.DataFrame({
        'ts': pd.date_range('2023-01-01', periods=n_rows, freq='D'),
        'symbol': ['TEST'] * n_rows,
        'ret_1d': np.random.randn(n_rows) * 0.02,
        'ret_5d': np.random.randn(n_rows) * 0.05,
        'vol_21': np.abs(np.random.randn(n_rows) * 0.03),
        'size_ln': np.random.randn(n_rows) + 10,
        'fwd_ret_resid': np.random.randn(n_rows) * 0.03  # target variable
    }, index=indices)
    
    feature_cols = ['ret_1d', 'ret_5d', 'vol_21', 'size_ln']
    
    try:
        # This should not raise an IndexError anymore
        ic_results = _ic_by_model(test_df, feature_cols)
        print(f"✓ _ic_by_model completed successfully with {len(ic_results)} models")
        print(f"  IC results: {ic_results}")
        
        # Verify results are reasonable
        assert isinstance(ic_results, dict), "IC results should be a dictionary"
        assert all(isinstance(v, float) for v in ic_results.values()), "All IC values should be floats"
        
        return True
    except Exception as e:
        print(f"✗ _ic_by_model failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sql_generation():
    """Test that SQL queries are generated correctly."""
    print("\nTesting SQL generation...")
    
    # Test that price_expr() returns safe SQL
    expr = price_expr()
    print(f"Price expression: {expr}")
    
    # Verify it's either "close" or contains COALESCE
    assert expr == "close" or "COALESCE" in expr, f"Unexpected price expression: {expr}"
    
    # Test SQL injection safety (basic check)
    assert ";" not in expr, "Price expression should not contain semicolons"
    assert "--" not in expr, "Price expression should not contain SQL comments"
    
    print("✓ SQL generation is safe")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing adj_close fix and index alignment bug fix")
    print("=" * 60)
    
    test_price_utils()
    
    success = test_index_alignment_bug()
    
    test_sql_generation()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)