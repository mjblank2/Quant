#!/usr/bin/env python3
"""Test the migration fixes for 20250910_03 and 20250910_04."""

import os
import tempfile
import sqlite3
from sqlalchemy import create_engine, text, inspect
import sqlalchemy as sa


def test_idempotent_migration():
    """Test that 20250910_03 migration is idempotent."""
    print("Testing idempotent migration 20250910_03...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        engine = create_engine(f'sqlite:///{db_path}')
        
        # Create features table with some existing columns
        with engine.connect() as conn:
            conn.execute(text('''
                CREATE TABLE features (
                    symbol TEXT NOT NULL,
                    ts DATE NOT NULL,
                    ret_1d REAL,
                    beta_63 REAL,
                    overnight_gap REAL,
                    PRIMARY KEY (symbol, ts)
                )
            '''))
            conn.commit()
            
            # Check initial state
            inspector = inspect(engine)
            initial_cols = {c['name'] for c in inspector.get_columns('features')}
            print(f"Initial columns: {sorted(initial_cols)}")
            
            # Apply our migration logic
            COLUMNS_TO_ADD = {
                'reversal_5d_z': sa.Float(),
                'ivol_63': sa.Float(),
                'beta_63': sa.Float(),
                'overnight_gap': sa.Float(),
                'illiq_21': sa.Float(),
                'fwd_ret': sa.Float(),
                'fwd_ret_resid': sa.Float(),
                'pead_event': sa.Float(),
                'pead_surprise_eps': sa.Float(),
                'pead_surprise_rev': sa.Float(),
                'russell_inout': sa.Float(),
            }
            
            existing = {c['name'] for c in inspector.get_columns('features')}
            added_cols = []
            skipped_cols = []
            
            for name, coltype in COLUMNS_TO_ADD.items():
                if name not in existing:
                    conn.execute(text(f'ALTER TABLE features ADD COLUMN {name} REAL'))
                    added_cols.append(name)
                else:
                    skipped_cols.append(name)
            
            conn.commit()
            
            print(f"Added columns: {sorted(added_cols)}")
            print(f"Skipped existing columns: {sorted(skipped_cols)}")
            
            # Verify final state
            inspector = inspect(engine)
            final_cols = {c['name'] for c in inspector.get_columns('features')}
            print(f"Final columns: {sorted(final_cols)}")
            
            # Check that all expected columns exist
            expected_cols = set(COLUMNS_TO_ADD.keys()) | initial_cols
            assert final_cols == expected_cols, f"Missing columns: {expected_cols - final_cols}"
            
            # Check that existing columns were skipped
            assert 'beta_63' in skipped_cols, "beta_63 should have been skipped"
            assert 'overnight_gap' in skipped_cols, "overnight_gap should have been skipped"
            
            print("✓ Idempotent migration test passed")
            
    finally:
        os.unlink(db_path)


def test_backfill_migration():
    """Test that 20250910_04 backfill migration works correctly."""
    print("\nTesting backfill migration 20250910_04...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        engine = create_engine(f'sqlite:///{db_path}')
        
        # Create features table with all columns
        with engine.connect() as conn:
            conn.execute(text('''
                CREATE TABLE features (
                    symbol TEXT NOT NULL,
                    ts DATE NOT NULL,
                    reversal_5d_z REAL,
                    ivol_63 REAL,
                    pead_event REAL,
                    pead_surprise_eps REAL,
                    pead_surprise_rev REAL,
                    russell_inout REAL,
                    fwd_ret REAL,
                    fwd_ret_resid REAL,
                    beta_63 REAL,
                    PRIMARY KEY (symbol, ts)
                )
            '''))
            
            # Insert test data with mixed NULL and non-NULL values
            conn.execute(text('''
                INSERT INTO features (symbol, ts, reversal_5d_z, pead_event, ivol_63, beta_63) 
                VALUES ('AAPL', '2023-01-01', NULL, NULL, 0.5, 1.2)
            '''))
            conn.execute(text('''
                INSERT INTO features (symbol, ts, reversal_5d_z, pead_event, ivol_63, beta_63) 
                VALUES ('TSLA', '2023-01-01', 1.5, NULL, NULL, NULL)
            '''))
            conn.commit()
            
            # Check before backfill
            result = conn.execute(text('''
                SELECT symbol, reversal_5d_z, pead_event, ivol_63, beta_63 
                FROM features ORDER BY symbol
            ''')).fetchall()
            print("Before backfill:")
            for row in result:
                print(f"  {row}")
            
            # Apply backfill logic
            inspector = inspect(engine)
            def col_exists(col: str) -> bool:
                existing = {c['name'] for c in inspector.get_columns('features')}
                return col in existing

            zero_cols = ['reversal_5d_z','ivol_63','pead_event','pead_surprise_eps','pead_surprise_rev','russell_inout']
            backfilled_cols = []
            
            for col in zero_cols:
                if col_exists(col):
                    conn.execute(text(f'UPDATE features SET {col}=0 WHERE {col} IS NULL'))
                    backfilled_cols.append(col)
            
            conn.commit()
            print(f"Backfilled columns: {sorted(backfilled_cols)}")
            
            # Check after backfill
            result = conn.execute(text('''
                SELECT symbol, reversal_5d_z, pead_event, ivol_63, beta_63 
                FROM features ORDER BY symbol
            ''')).fetchall()
            print("After backfill:")
            for row in result:
                print(f"  {row}")
            
            # Verify backfill behavior
            aapl_row = [r for r in result if r[0] == 'AAPL'][0]
            tsla_row = [r for r in result if r[0] == 'TSLA'][0]
            
            # AAPL: reversal_5d_z NULL->0, pead_event NULL->0, ivol_63 0.5->0.5, beta_63 1.2->1.2
            assert aapl_row[1] == 0.0, f"AAPL reversal_5d_z should be 0, got {aapl_row[1]}"
            assert aapl_row[2] == 0.0, f"AAPL pead_event should be 0, got {aapl_row[2]}"
            assert aapl_row[3] == 0.5, f"AAPL ivol_63 should remain 0.5, got {aapl_row[3]}"
            assert aapl_row[4] == 1.2, f"AAPL beta_63 should remain 1.2, got {aapl_row[4]}"
            
            # TSLA: reversal_5d_z 1.5->1.5, pead_event NULL->0, ivol_63 NULL->0, beta_63 NULL->NULL
            assert tsla_row[1] == 1.5, f"TSLA reversal_5d_z should remain 1.5, got {tsla_row[1]}"
            assert tsla_row[2] == 0.0, f"TSLA pead_event should be 0, got {tsla_row[2]}"
            assert tsla_row[3] == 0.0, f"TSLA ivol_63 should be 0, got {tsla_row[3]}"
            # beta_63 is NOT in zero_cols, so should remain NULL
            assert tsla_row[4] is None, f"TSLA beta_63 should remain NULL, got {tsla_row[4]}"
            
            print("✓ Backfill migration test passed")
            
    finally:
        os.unlink(db_path)


if __name__ == "__main__":
    test_idempotent_migration()
    test_backfill_migration()
    print("\n🎉 All migration tests passed!")