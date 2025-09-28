#!/usr/bin/env python3
"""
Test to reproduce the exact parameter limit issue from the problem statement.
"""

import os
import tempfile
import pandas as pd
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

def test_large_universe_insert():
    """Test inserting a large number of universe records to reproduce the parameter limit issue."""
    print("🧪 Testing large universe insert to reproduce parameter limit issue...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
        
        # Import after setting DATABASE_URL
        import db
        from db import Universe
        
        # Create database tables
        db.Base.metadata.create_all(db.engine)
        
        # Create a dataset large enough to potentially cause parameter limit issues
        # Try with different sizes to find the breaking point
        test_sizes = [250, 300, 500, 1000]
        
        for size in test_sizes:
            print(f"\n🔄 Testing with {size} records...")
            
            # Create DataFrame with universe data
            df_data = []
            for i in range(size):
                df_data.append({
                    "symbol": f"TST{i:04d}",
                    "name": f"Test Company {i}",
                    "included": True,
                    "last_updated": datetime.utcnow(),
                })
            
            df = pd.DataFrame(df_data)
            print(f"   DataFrame created with {len(df)} records, {len(df.columns)} columns")
            param_count = len(df) * len(df.columns)
            print(f"   Theoretical parameters needed: {param_count} = {param_count}")
            print(f"   SQLite parameter limit: 999")
            
            try:
                # Use upsert_dataframe directly (this is what rebuild_universe uses)
                db.upsert_dataframe(df, Universe, conflict_cols=["symbol"])
                print(f"   ✅ Successfully inserted {size} records")
                
                # Verify insertion
                with db.engine.connect() as conn:
                    count = conn.execute(db.text("SELECT COUNT(*) FROM universe")).scalar()
                    print(f"   📊 Database now contains {count} universe records")
                    
            except Exception as e:
                print(f"   ❌ Failed with {size} records: {e}")
                
                # Check if this looks like the parameter limit error from the problem statement
                error_str = str(e).lower()
                if "too many sql variables" in error_str or "too many variables" in error_str:
                    print(f"   🎯 This matches the parameter limit error pattern!")
                    return True
                else:
                    print(f"   🤔 Different error type: {error_str}")
        
        print("\n✅ All test sizes completed without hitting parameter limits")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        try:
            os.unlink(db_path)
        except:
            pass

def test_direct_sqlalchemy_insert():
    """Test direct SQLAlchemy insert to see parameter usage pattern."""
    print("\n🧪 Testing direct SQLAlchemy insert to understand parameter usage...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
        
        import db
        from db import Universe
        from sqlalchemy.dialects import sqlite
        
        # Create database tables
        db.Base.metadata.create_all(db.engine)
        
        # Test with exactly 250 records (which should be safe: 250 * 4 = 1000 > 999)
        records = []
        for i in range(250):
            records.append({
                "symbol": f"TST{i:04d}",
                "name": f"Test Company {i}",
                "included": True,
                "last_updated": datetime.utcnow(),
            })
        
        print(f"   Testing with {len(records)} records")
        print(f"   Expected parameters: {len(records) * 4} = {len(records) * 4}")
        
        with db.engine.connect() as conn:
            from sqlalchemy import insert
            
            # This is exactly what upsert_dataframe does
            stmt = insert(Universe).values(records)
            
            # Print the compiled SQL to see parameter pattern
            compiled = stmt.compile(dialect=sqlite.dialect(), compile_kwargs={"literal_binds": False})
            sql_str = str(compiled)
            
            print(f"   📝 Generated SQL length: {len(sql_str)} characters")
            
            # Count parameter placeholders
            import re
            params = re.findall(r'%\([^)]+\)', sql_str)
            print(f"   🔢 Parameter count in SQL: {len(params)}")
            
            if len(params) > 999:
                print(f"   🎯 Found the issue! {len(params)} parameters > 999 SQLite limit")
                print(f"   📄 First few parameters: {params[:10]}")
                print(f"   📄 Last few parameters: {params[-10:]}")
                
                # Show part of the SQL that demonstrates the pattern
                if "symbol_m" in sql_str:
                    print(f"   📄 SQL contains the pattern from the error message!")
                
                return True
            else:
                print(f"   ✅ Parameter count {len(params)} is within SQLite limit")
                
                # Try to execute it
                try:
                    result = conn.execute(stmt)
                    print(f"   ✅ Execution successful")
                    return True
                except Exception as e:
                    print(f"   ❌ Execution failed: {e}")
                    return False
        
    except Exception as e:
        print(f"❌ Direct SQLAlchemy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            os.unlink(db_path)
        except:
            pass

if __name__ == "__main__":
    print("🚀 Starting parameter limit issue reproduction...")
    
    test1_success = test_large_universe_insert()
    test2_success = test_direct_sqlalchemy_insert()
    
    if test1_success and test2_success:
        print("\n✅ Parameter limit reproduction tests completed!")
        exit(0)
    else:
        print("\n❌ Some parameter limit reproduction tests failed!")
        exit(1)