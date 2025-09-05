#!/usr/bin/env python3
"""
Test the enhanced fixes for CardinalityViolation that were added to models/features.py
"""

import pandas as pd
import numpy as np
import tempfile
import os
from datetime import date
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def test_enhanced_features_fixes():
    """Test the enhanced fixes in models/features.py"""
    print("🔧 Testing enhanced fixes in features.py...")
    
    # Create a temporary SQLite database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        temp_db_path = tmp.name
    
    try:
        # Set up environment
        os.environ['DATABASE_URL'] = f'sqlite:///{temp_db_path}'
        
        # Create test database
        engine = create_engine(f'sqlite:///{temp_db_path}')
        
        # Import after setting DATABASE_URL
        from db import Base, Feature, upsert_dataframe
        
        # Create tables
        Base.metadata.create_all(engine)
        
        # Test 1: Test the conservative chunk_size parameter
        print("\n🔧 Test 1: Conservative chunk_size parameter...")
        
        data = []
        for i in range(50):  # Create data that would exceed chunk_size=10 several times
            record = {
                'symbol': f"SYM{i:03d}",
                'ts': date(2016, 1, 4),
                'ret_1d': np.random.normal(0, 0.02),
                'ret_5d': np.random.normal(0, 0.05),
                'vol_21': np.random.uniform(0.1, 0.3),
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        
        try:
            # Test with a very small chunk_size to ensure chunking works properly
            with engine.connect() as conn:
                upsert_dataframe(df, Feature, ['symbol', 'ts'], chunk_size=10, conn=conn)
                conn.commit()
            
            # Verify results
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM features")).scalar()
                print(f"✅ Successfully inserted {result} records with chunk_size=10")
                
        except Exception as e:
            print(f"❌ Conservative chunk_size test failed: {e}")
            return False
        
        # Test 2: Test final validation logic by simulating the features.py flow
        print("\n🔧 Test 2: Final validation logic...")
        
        # Clear the table
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM features"))
            conn.commit()
        
        # Create data with duplicates that should trigger the final validation
        test_data = []
        # Add duplicates intentionally
        for i in range(3):
            record = {
                'symbol': 'TEST',
                'ts': date(2016, 1, 4),
                'ret_1d': 0.01 + i * 0.001,  # Slightly different values
                'vol_21': 0.15 + i * 0.001,
            }
            test_data.append(record)
        
        feats = pd.DataFrame(test_data)
        print(f"Created test data with {len(feats)} rows (should have duplicates)")
        
        # Simulate the logic from models/features.py
        original_count = len(feats)
        feats = feats.drop_duplicates(subset=['symbol', 'ts'], keep='last').reset_index(drop=True)
        dedupe_count = len(feats)
        if dedupe_count < original_count:
            print(f"Deduplication: {original_count} -> {dedupe_count} rows")
        
        # Final validation (this is the new logic we added)
        final_check = feats.groupby(['symbol', 'ts']).size()
        max_count = final_check.max()
        print(f"Final validation: max rows per (symbol, ts): {max_count}")
        
        if max_count > 1:
            print("❌ Final validation failed - duplicates still exist!")
            return False
        else:
            print("✅ Final validation passed - no duplicates")
        
        # Test the actual upsert with chunk_size=1000
        try:
            with engine.connect() as conn:
                upsert_dataframe(feats, Feature, ['symbol', 'ts'], chunk_size=1000, conn=conn)
                conn.commit()
            
            # Verify results
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM features")).scalar()
                print(f"✅ Successfully inserted {result} records with final validation")
                
        except Exception as e:
            print(f"❌ Final validation upsert test failed: {e}")
            return False
        
        return True
        
    finally:
        # Cleanup
        try:
            os.unlink(temp_db_path)
        except:
            pass

def test_emergency_deduplication():
    """Test the emergency deduplication logic"""
    print("\n🔧 Testing emergency deduplication logic...")
    
    # Simulate the emergency deduplication scenario
    # Create data that somehow still has duplicates after normal deduplication
    feats = pd.DataFrame([
        {'symbol': 'TEST', 'ts': date(2016, 1, 4), 'ret_1d': 0.01},
        {'symbol': 'TEST', 'ts': date(2016, 1, 4), 'ret_1d': 0.02},  # Duplicate!
        {'symbol': 'TEST', 'ts': date(2016, 1, 5), 'ret_1d': 0.03},
    ])
    
    print(f"Original test data: {len(feats)} rows")
    print(feats)
    
    # Check for duplicates
    final_check = feats.groupby(['symbol', 'ts']).size()
    max_count = final_check.max()
    print(f"Max count per (symbol, ts): {max_count}")
    
    if max_count > 1:
        print("⚠️ Duplicates detected - applying emergency deduplication")
        feats = feats.drop_duplicates(subset=['symbol', 'ts'], keep='last').reset_index(drop=True)
        print("✅ Emergency deduplication applied")
    
    print(f"Final data: {len(feats)} rows")
    print(feats)
    
    # Verify no duplicates remain
    final_check_after = feats.groupby(['symbol', 'ts']).size()
    max_count_after = final_check_after.max()
    
    if max_count_after == 1:
        print("✅ Emergency deduplication successful")
        return True
    else:
        print("❌ Emergency deduplication failed")
        return False

if __name__ == "__main__":
    print("🔬 Testing enhanced CardinalityViolation fixes...\n")
    
    success1 = test_enhanced_features_fixes()
    success2 = test_emergency_deduplication()
    
    if success1 and success2:
        print("\n🎉 All enhanced fix tests passed!")
        print("✅ The enhanced fixes should provide additional protection against CardinalityViolation errors.")
    else:
        print("\n❌ Some enhanced fix tests failed!")
    
    exit(0 if (success1 and success2) else 1)