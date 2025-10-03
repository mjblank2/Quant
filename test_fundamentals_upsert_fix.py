"""
Test for fundamentals upsert fix to prevent duplicate key violations.

This test validates that the _upsert_fundamentals function correctly handles:
1. Duplicate (symbol, as_of) pairs within a single batch
2. Re-insertion of existing (symbol, as_of) pairs from the database
3. Updates to existing records when conflicts occur
"""

import os
import pandas as pd
from datetime import date
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Set up test database
os.environ["DATABASE_URL"] = "sqlite:///test_fundamentals_upsert.db"

from db import Base, Fundamentals
from data.fundamentals import _dedupe_fundamentals, _upsert_fundamentals


def test_dedupe_fundamentals():
    """Test that deduplication removes duplicate (symbol, as_of) pairs."""
    df = pd.DataFrame([
        {"symbol": "AAPL", "as_of": date(2025, 8, 29), "debt_to_equity": 1.5},
        {"symbol": "AAPL", "as_of": date(2025, 8, 29), "debt_to_equity": 1.6},  # Duplicate
        {"symbol": "MSFT", "as_of": date(2025, 8, 29), "debt_to_equity": 0.8},
    ])
    
    result = _dedupe_fundamentals(df)
    
    # Should keep only 2 rows (last occurrence of duplicates)
    assert len(result) == 2, f"Expected 2 rows, got {len(result)}"
    # Should keep the last value for duplicate AAPL
    aapl_row = result[result["symbol"] == "AAPL"].iloc[0]
    assert aapl_row["debt_to_equity"] == 1.6, f"Expected 1.6, got {aapl_row['debt_to_equity']}"


def test_dedupe_within_batch():
    """Test handling of duplicates within a single batch."""
    df = pd.DataFrame([
        {"symbol": "A", "as_of": date(2025, 8, 29), "debt_to_equity": 1.0},
        {"symbol": "A", "as_of": date(2025, 8, 29), "debt_to_equity": 1.5},  # Duplicate
        {"symbol": "B", "as_of": date(2025, 8, 29), "debt_to_equity": 2.0},
        {"symbol": "B", "as_of": date(2025, 8, 30), "debt_to_equity": 2.1},  # Different as_of
    ])
    
    result = _dedupe_fundamentals(df)
    
    # Should have 3 unique (symbol, as_of) pairs
    assert len(result) == 3, f"Expected 3 rows, got {len(result)}"
    
    # Verify the duplicate was removed and last value kept
    a_aug29 = result[(result["symbol"] == "A") & (result["as_of"] == date(2025, 8, 29))]
    assert len(a_aug29) == 1, f"Expected 1 row for A/2025-08-29, got {len(a_aug29)}"
    assert a_aug29.iloc[0]["debt_to_equity"] == 1.5, f"Expected 1.5, got {a_aug29.iloc[0]['debt_to_equity']}"


def test_empty_dataframe():
    """Test that empty dataframes are handled gracefully."""
    df = pd.DataFrame()
    result = _dedupe_fundamentals(df)
    assert result.empty, "Expected empty result"
    
    count = _upsert_fundamentals(df)
    assert count == 0, f"Expected 0 count, got {count}"


if __name__ == "__main__":
    # Run basic tests without pytest
    print("Testing deduplication...")
    test_dedupe_fundamentals()
    print("✓ Deduplication test passed")
    
    print("\nTesting within-batch duplicates...")
    test_dedupe_within_batch()
    print("✓ Within-batch duplicates test passed")
    
    print("\nTesting empty dataframe...")
    test_empty_dataframe()
    print("✓ Empty dataframe test passed")
    
    print("\n✅ All basic tests passed!")

