# Fundamentals Upsert Fix - Quick Reference

## Problem Fixed
```
psycopg.errors.UniqueViolation: duplicate key value violates unique constraint "fundamentals_pkey"
DETAIL: Key (symbol, as_of)=(A, 2025-08-29) already exists.
```

## Solution Summary
Implemented PostgreSQL `ON CONFLICT DO UPDATE` for the fundamentals table to handle duplicate primary keys gracefully.

## Quick Test
```bash
# Run all validation tests
python run_all_fundamentals_tests.py

# Or run individual tests
python test_fundamentals_upsert_fix.py          # Unit tests
python test_fundamentals_integration.py         # Integration tests  
python demo_fundamentals_fix.py                 # Interactive demo
```

## What Changed

### Core Files (80 lines)
- **`data/fundamentals.py`**: Added `_upsert_fundamentals()` with ON CONFLICT logic
- **`data/institutional_ingest.py`**: Updated to use new upsert function

### Key Functions Added
```python
def _dedupe_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate (symbol, as_of) pairs within batch."""
    
def _upsert_fundamentals(df: pd.DataFrame, chunk_size: int = 1000) -> int:
    """Upsert using PostgreSQL ON CONFLICT DO UPDATE."""
```

## How It Works
1. **Deduplicates** incoming data on (symbol, as_of), keeping last occurrence
2. **Filters** to valid table columns, drops null primary keys
3. **Chunks** large batches (1000 rows) to avoid parameter limits
4. **Uses ON CONFLICT** to update existing records instead of failing:
   ```sql
   INSERT INTO fundamentals (symbol, as_of, ...)
   VALUES (...)
   ON CONFLICT (symbol, as_of) DO UPDATE
   SET debt_to_equity = EXCLUDED.debt_to_equity, ...
   ```

## Testing
All tests pass ✅:
- Unit tests for deduplication logic
- Integration tests for duplicate key scenarios
- Manual validation demo

## Production Impact
✅ No more UniqueViolation errors  
✅ Existing records are updated with new data  
✅ Within-batch duplicates handled automatically  
✅ No breaking changes  

## Documentation
See `FUNDAMENTALS_UPSERT_FIX.md` for complete implementation details.

## Related Files
- Implementation: `data/fundamentals.py`, `data/institutional_ingest.py`
- Tests: `test_fundamentals_*.py`, `demo_fundamentals_fix.py`
- Test runner: `run_all_fundamentals_tests.py`
- Docs: `FUNDAMENTALS_UPSERT_FIX.md`
