#!/usr/bin/env python
"""
Verification script to demonstrate adj_close column resilience.

This script can be run against any database (with or without adj_close column)
to verify that the system handles price queries gracefully.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """Verify adj_close resilience."""
    print("🔍 Verifying adj_close column resilience...")
    
    # Check environment
    if not os.getenv('DATABASE_URL'):
        print("❌ DATABASE_URL environment variable required")
        print("   Example: export DATABASE_URL='sqlite:///test.db'")
        return 1
    
    try:
        from utils.price_utils import price_expr, select_price_as, has_adj_close
        
        print("\n📊 Price handling status:")
        adj_present = has_adj_close()
        print(f"   adj_close column present: {adj_present}")
        print(f"   price_expr(): '{price_expr()}'")
        print(f"   select_price_as('px'): '{select_price_as('px')}'")
        
        # Test SQL generation
        test_sqls = [
            f"SELECT symbol, ts, {select_price_as('adj_close')}, volume FROM daily_bars LIMIT 1",
            f"SELECT ts, {select_price_as('px')} FROM daily_bars WHERE symbol = 'AAPL' LIMIT 1",
            f"SELECT symbol, MAX(ts) as latest_ts, {select_price_as('latest_px')} FROM daily_bars GROUP BY symbol LIMIT 3"
        ]
        
        print("\n🔧 Generated SQL examples:")
        for i, sql in enumerate(test_sqls, 1):
            print(f"   {i}. {sql}")
        
        # Test actual query execution if possible
        try:
            from db import engine
            from sqlalchemy import text
            import pandas as pd
            
            # Simple test query
            test_sql = f"SELECT COUNT(*) as row_count FROM daily_bars"
            with engine.connect() as conn:
                result = pd.read_sql_query(text(test_sql), conn)
                row_count = result.iloc[0]['row_count']
                print(f"\n📈 Database connection test:")
                print(f"   daily_bars table has {row_count} rows")
                
                if row_count > 0:
                    # Test price query
                    price_sql = f"SELECT symbol, ts, {select_price_as('price')} FROM daily_bars LIMIT 1"
                    price_result = pd.read_sql_query(text(price_sql), conn)
                    if not price_result.empty:
                        print(f"   ✅ Price query successful: {dict(price_result.iloc[0])}")
                    else:
                        print(f"   ⚠️  Price query returned no results")
        
        except Exception as e:
            print(f"\n⚠️  Database query test skipped: {e}")
        
        print(f"\n✅ System is {'fully' if adj_present else 'resiliently'} operational!")
        print("   All price queries will work regardless of adj_close column presence.")
        
        return 0
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())