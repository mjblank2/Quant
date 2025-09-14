#!/usr/bin/env python3
"""
Demonstration that the parameter limit fix resolves the original cronjob issue.

This simulates the exact scenario described in the problem statement:
- Bulk inserts with many parameters (like %(symbol_m785)s::VARCHAR)
- Proper chunking and retry when parameter limits are exceeded
- No more "Exited with status 1" failures due to parameter limit errors
"""

import pandas as pd
import numpy as np
import logging
import tempfile
import os
from datetime import date, datetime
from unittest.mock import patch

# Setup logging similar to cronjob environment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def simulate_cronjob_scenario():
    """Simulate the scenario that was causing cronjob failures."""
    
    print("🚀 Simulating cronjob scenario that previously caused 'Exited with status 1'...")
    
    # Set up test database
    db_file = tempfile.mktemp(suffix='.db')
    test_db_url = f"sqlite:///{db_file}"
    
    try:
        # Set environment like cronjob would
        os.environ['DATABASE_URL'] = test_db_url
        
        # Import db module (like pipeline would)
        import db
        
        # Create tables (like migrations would)
        db.create_tables()
        log.info("✅ Database tables created successfully")
        
        # Simulate the universe rebuild operation that often causes parameter limit issues
        print("\n📊 Simulating universe rebuild with many symbols...")
        
        # Create a large universe dataset (like what triggers the parameter errors)
        universe_data = []
        for i in range(2000):  # Large dataset like real universe
            universe_data.append({
                'symbol': f'STOCK{i:04d}',
                'name': f'Company {i} Inc.',
                'exchange': 'NASDAQ' if i % 2 == 0 else 'NYSE',
                'market_cap': 1000000.0 + i * 10000,
                'adv_usd_20': 50000.0 + i * 500,
                'included': True,
                'last_updated': datetime.now()
            })
        
        universe_df = pd.DataFrame(universe_data)
        log.info(f"Created universe DataFrame: {len(universe_df)} rows × {len(universe_df.columns)} columns")
        log.info(f"Total parameters: {len(universe_df) * len(universe_df.columns)}")
        
        # Test bulk upsert that would previously cause parameter limit failures
        conflict_cols = ['symbol']
        
        # Use large chunk size to potentially trigger parameter limits
        log.info("Performing bulk upsert with large chunk size...")
        db.upsert_dataframe(universe_df, db.Universe, conflict_cols, chunk_size=50000)
        log.info("✅ Universe bulk upsert completed successfully")
        
        # Simulate feature engineering bulk insert (another common failure point)
        print("\n🧮 Simulating feature engineering bulk insert...")
        
        feature_data = []
        symbols = [f'STOCK{i:04d}' for i in range(200)]  # Subset for features
        
        for symbol in symbols:
            feature_data.append({
                'symbol': symbol,
                'ts': date(2024, 1, 15),
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
            })
        
        features_df = pd.DataFrame(feature_data)
        log.info(f"Created features DataFrame: {len(features_df)} rows × {len(features_df.columns)} columns")
        log.info(f"Total parameters: {len(features_df) * len(features_df.columns)}")
        
        # This type of operation frequently caused the parameter limit errors
        conflict_cols = ['symbol', 'ts']
        db.upsert_dataframe(features_df, db.Feature, conflict_cols, chunk_size=50000)
        log.info("✅ Features bulk upsert completed successfully")
        
        # Verify final state
        with db.engine.connect() as conn:
            universe_count = conn.execute(db.text('SELECT COUNT(*) FROM universe')).scalar()
            features_count = conn.execute(db.text('SELECT COUNT(*) FROM features')).scalar()
            
        log.info(f"✅ Final verification:")
        log.info(f"   - Universe records: {universe_count}")
        log.info(f"   - Feature records: {features_count}")
        
        print("\n🎉 Cronjob scenario simulation completed successfully!")
        print("   ✅ No parameter limit errors")
        print("   ✅ All bulk operations completed")
        print("   ✅ Exit status: 0 (success)")
        
        return True
        
    except Exception as e:
        log.error(f"❌ Simulation failed: {e}")
        print(f"\n💥 Cronjob scenario simulation failed!")
        print(f"   ❌ Error: {e}")
        print(f"   ❌ Exit status: 1 (failure)")
        return False
        
    finally:
        # Clean up
        try:
            if os.path.exists(db_file):
                os.unlink(db_file)
        except:
            pass

if __name__ == "__main__":
    success = simulate_cronjob_scenario()
    exit(0 if success else 1)