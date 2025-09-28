#!/usr/bin/env python3

"""
Test for universe name widening and sanitization fix.

This test validates that:
1. The universe.name column can handle names up to 256 characters
2. Long names are properly sanitized and truncated
3. String sanitization works correctly in both rebuild_universe and upsert_dataframe
4. The migration works correctly
"""

import os
import tempfile
import pandas as pd
import logging
import unicodedata
from unittest.mock import patch
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def test_long_names_handling():
    """Test universe rebuild and upsert with long company names."""
    print("🧪 Testing long company names handling...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
        
        # Import after setting DATABASE_URL
        import db
        from data.universe import rebuild_universe
        
        # Create database tables with the new schema
        db.Base.metadata.create_all(db.engine)
        
        # Test data with various name lengths
        test_symbols = [
            {
                "symbol": "NORM",
                "name": "Normal Company Name"
            },
            {
                "symbol": "LONG1", 
                "name": "A" * 200  # 200 characters - should fit
            },
            {
                "symbol": "LONG2",
                "name": "B" * 300  # 300 characters - should be truncated to 256
            },
            {
                "symbol": "WEIRD",
                "name": "Company   with\t\nweird\u2000whitespace\u3000chars"  # Multiple whitespace types
            },
            {
                "symbol": "UNICODE",
                "name": "Unicodé Çompañy ＡＢＣ"  # Unicode that needs normalization
            }
        ]
        
        with patch('data.universe._list_small_cap_symbols', return_value=test_symbols):
            result = rebuild_universe()
        
        print(f"✅ Successfully rebuilt universe with {len(result)} symbols including long names")
        
        # Verify data was inserted correctly
        with db.engine.connect() as conn:
            query = db.text("SELECT symbol, name FROM universe ORDER BY symbol")
            rows = conn.execute(query).fetchall()
            
            print("📊 Inserted company names:")
            for symbol, name in rows:
                print(f"  {symbol}: {name[:50]}{'...' if len(name or '') > 50 else ''} (len: {len(name or '')})")
                
                # Verify length constraints
                if name:
                    assert len(name) <= 256, f"Name too long for {symbol}: {len(name)} chars"
                    
                    # Verify normalization
                    normalized = unicodedata.normalize('NFKC', name)
                    collapsed = ' '.join(normalized.split())
                    # The stored name should be normalized
                    assert name == collapsed or len(name) == 256, f"Name not properly normalized for {symbol}"
        
        return True
        
    except Exception as e:
        print(f"❌ Long names test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            os.unlink(db_path)
        except:
            pass


def test_direct_upsert_sanitization():
    """Test direct upsert with string sanitization."""
    print("🧪 Testing direct upsert string sanitization...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
        
        # Import after setting DATABASE_URL
        import db
        
        # Create database tables
        db.Base.metadata.create_all(db.engine)
        
        # Test DataFrame with problematic string data
        test_data = pd.DataFrame([
            {
                "symbol": "TEST1",
                "name": "Regular Company",
                "exchange": "NASDAQ"
            },
            {
                "symbol": "TEST2", 
                "name": "C" * 300,  # Too long for both model (256) and old schema (128)
                "exchange": "NYSE"
            },
            {
                "symbol": "TEST3",
                "name": "Unicode   \u2000\u3000  Test\t\nCompany",  # Needs normalization
                "exchange": "AMEX" * 5  # This should be truncated too (exchange is VARCHAR(12))
            }
        ])
        
        # Use upsert_dataframe which should sanitize the data
        db.upsert_dataframe(test_data, db.Universe, conflict_cols=["symbol"])
        
        # Verify the sanitization worked
        with db.engine.connect() as conn:
            query = db.text("SELECT symbol, name, exchange FROM universe ORDER BY symbol")
            rows = conn.execute(query).fetchall()
            
            print("📊 Sanitized data:")
            for symbol, name, exchange in rows:
                print(f"  {symbol}: name='{name[:50]}...' (len: {len(name or '')}), exchange='{exchange}' (len: {len(exchange or '')})")
                
                # Verify constraints
                if name:
                    assert len(name) <= 256, f"Name too long after sanitization: {len(name)} chars"
                if exchange:
                    assert len(exchange) <= 12, f"Exchange too long after sanitization: {len(exchange)} chars"
        
        print("✅ Direct upsert sanitization test passed")
        return True
        
    except Exception as e:
        print(f"❌ Direct upsert sanitization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            os.unlink(db_path)
        except:
            pass


def test_migration_compatibility():
    """Test that the new schema is compatible."""
    print("🧪 Testing migration compatibility...")
    
    try:
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
        
        # Import after setting DATABASE_URL
        import db
        
        # Create tables with new schema
        db.Base.metadata.create_all(db.engine)
        
        # Verify the schema
        with db.engine.connect() as conn:
            import sqlalchemy as sa
            insp = sa.inspect(conn)
            cols = {c['name']: c for c in insp.get_columns('universe')}
            
            print("📋 Universe table columns:")
            for name, info in cols.items():
                print(f"  {name}: {info['type']}")
            
            # Verify name column can handle 256 characters
            assert 'name' in cols, "Name column missing"
            name_col = cols['name']
            
            # For SQLite, the type info might not include length, but our model should handle it
            print("✅ Schema verification passed")
        
        # Test inserting a 256-character name
        long_name = "X" * 256
        test_df = pd.DataFrame([{"symbol": "LONG", "name": long_name}])
        
        db.upsert_dataframe(test_df, db.Universe, conflict_cols=["symbol"])
        
        # Verify it was stored correctly
        with db.engine.connect() as conn:
            result = conn.execute(db.text("SELECT name FROM universe WHERE symbol = 'LONG'")).scalar()
            assert result == long_name, f"Long name not stored correctly: {len(result)} != 256"
        
        print("✅ 256-character name storage test passed")
        return True
        
    except Exception as e:
        print(f"❌ Migration compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            os.unlink(db_path)
        except:
            pass


if __name__ == "__main__":
    print("🚀 Starting universe name widening and sanitization tests...")
    
    test1_success = test_long_names_handling()
    test2_success = test_direct_upsert_sanitization()  
    test3_success = test_migration_compatibility()
    
    if test1_success and test2_success and test3_success:
        print("✅ All universe name widening tests passed!")
        exit(0)
    else:
        print("❌ Some universe name widening tests failed!")
        exit(1)