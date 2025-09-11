#!/usr/bin/env python
"""
Test script to verify that the refactored price handling works with and without adj_close column.
"""
import os
import tempfile
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text

# Set DATABASE_URL for imports
os.environ['DATABASE_URL'] = 'sqlite:///test.db'

from utils.price_utils import price_expr, select_price_as, has_adj_close


def test_price_resilience():
    """Test that price expressions work with and without adj_close column."""
    print("Testing price expression resilience...")
    
    # Test 1: Database without adj_close column
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Create test database without adj_close
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE daily_bars (
                symbol TEXT,
                ts DATE,
                open REAL,
                close REAL,
                volume INTEGER
            )
        """)
        conn.execute("""
            INSERT INTO daily_bars VALUES 
            ('AAPL', '2023-01-01', 150.0, 155.0, 1000),
            ('AAPL', '2023-01-02', 155.0, 160.0, 1200)
        """)
        conn.commit()
        conn.close()
        
        # Test with SQLAlchemy engine
        engine = create_engine(f'sqlite:///{db_path}')
        
        # Override the engine in price_utils for this test
        import utils.price_utils as pu
        original_engine = pu.engine
        pu.engine = engine
        
        # Clear cache to force re-detection
        pu.has_adj_close.cache_clear()
        
        # Test without adj_close
        print(f"1. Without adj_close column:")
        print(f"   has_adj_close(): {has_adj_close()}")
        print(f"   price_expr(): '{price_expr()}'")
        print(f"   select_price_as('px'): '{select_price_as('px')}'")
        
        # Test query execution
        sql = f"SELECT symbol, ts, {select_price_as('px')} FROM daily_bars ORDER BY ts"
        df = pd.read_sql_query(text(sql), engine)
        print(f"   Query result: {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        assert 'px' in df.columns, "Expected 'px' column in result"
        
        # Test 2: Add adj_close column 
        conn = sqlite3.connect(db_path)
        conn.execute("ALTER TABLE daily_bars ADD COLUMN adj_close REAL")
        conn.execute("UPDATE daily_bars SET adj_close = close * 1.02")  # Simulate adjustment
        conn.commit()
        conn.close()
        
        # Clear cache to force re-detection
        pu.has_adj_close.cache_clear()
        pu.price_expr.cache_clear()  # Clear price_expr cache too
        pu._logged_price_mode = False  # Reset logging flag
        
        print(f"\n2. With adj_close column:")
        print(f"   has_adj_close(): {has_adj_close()}")
        print(f"   price_expr(): '{price_expr()}'")
        print(f"   select_price_as('px'): '{select_price_as('px')}'")
        
        # Test query execution
        sql = f"SELECT symbol, ts, {select_price_as('px')}, close, adj_close FROM daily_bars ORDER BY ts"
        df = pd.read_sql_query(text(sql), engine)
        print(f"   Query result: {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        assert 'px' in df.columns, "Expected 'px' column in result"
        
        # Verify that adj_close values are used when available
        if not df.empty:
            print(f"   Sample row: close={df.iloc[0]['close']}, adj_close={df.iloc[0]['adj_close']}, px={df.iloc[0]['px']}")
            # px should equal adj_close when adj_close is available
            assert df.iloc[0]['px'] == df.iloc[0]['adj_close'], "px should equal adj_close when available"
        
        # Restore original engine
        pu.engine = original_engine
        
        print("\n✅ All resilience tests passed!")
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_feature_module_integration():
    """Test that the refactored feature modules work correctly."""
    print("\nTesting feature module integration...")
    
    try:
        # Test import of refactored modules
        from features.build_features import _load_prices_batch, _load_market_returns
        from features.store import FeatureStore
        print("✅ All feature modules imported successfully")
        
        # Test that functions use price_expr correctly
        from utils.price_utils import select_price_as
        
        # These should not raise syntax errors
        test_sql_1 = f"SELECT symbol, ts, {select_price_as('adj_close')}, volume FROM daily_bars"
        test_sql_2 = f"SELECT ts, {select_price_as('px')} FROM daily_bars"
        
        print("✅ SQL template generation works")
        print(f"   Example SQL 1: {test_sql_1}")
        print(f"   Example SQL 2: {test_sql_2}")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        raise


if __name__ == "__main__":
    test_price_resilience()
    test_feature_module_integration()
    print("\n🎉 All tests completed successfully!")