# Fundamentals Duplicate Key Violation Fix - Implementation Summary

## Problem Statement

The EOD pipeline was failing with a PostgreSQL UniqueViolation error:

```
psycopg.errors.UniqueViolation: duplicate key value violates unique constraint "fundamentals_pkey"
DETAIL: Key (symbol, as_of)=(A, 2025-08-29) already exists.
```

This occurred during the fundamentals ingestion step when the pipeline attempted to insert records with primary key combinations `(symbol, as_of)` that already existed in the database.

## Root Cause Analysis

The issue was in the `upsert_dataframe()` function in `db.py`, which uses:

```python
df.to_sql(table_name, con=engine, if_exists="append", ...)
```

This performs a plain SQL INSERT without any conflict handling. When records with duplicate primary keys already exist in the database, PostgreSQL raises a UniqueViolation error.

The fundamentals table has a composite primary key on `(symbol, as_of)`:

```python
class Fundamentals(Base):
    __tablename__ = "fundamentals"
    symbol = Column(String, primary_key=True, index=True, nullable=False)
    as_of = Column(Date, primary_key=True, index=True, nullable=False)
    # ... other columns
```

## Solution Implementation

### 1. Created Dedicated Upsert Function

Added `_upsert_fundamentals()` in `data/fundamentals.py` that uses PostgreSQL's ON CONFLICT syntax:

```python
def _upsert_fundamentals(df: pd.DataFrame, chunk_size: int = 1000) -> int:
    """Upsert to fundamentals using PostgreSQL ON CONFLICT."""
    # 1. Deduplicate within-batch duplicates
    df = _dedupe_fundamentals(df)
    
    # 2. Filter to valid columns and drop null primary keys
    # ... 
    
    # 3. Use ON CONFLICT DO UPDATE for each chunk
    with engine.begin() as conn:
        for start_idx in range(0, len(df), chunk_size):
            chunk = df.iloc[start_idx:start_idx + chunk_size]
            payload = chunk.to_dict(orient="records")
            stmt = pg_insert(table).values(payload)
            
            # Build update dict for all non-key columns
            update_dict = {}
            for col in df_cols:
                if col not in ["symbol", "as_of"]:
                    update_dict[col] = stmt.excluded[col]
            
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "as_of"],
                set_=update_dict,
            )
            conn.execute(stmt)
```

### 2. Added Deduplication

Created `_dedupe_fundamentals()` to handle within-batch duplicates:

```python
def _dedupe_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate (symbol, as_of) to avoid ON CONFLICT errors."""
    if df is None or df.empty:
        return df
    before = len(df)
    df = df.sort_values(["symbol", "as_of"]).drop_duplicates(
        subset=["symbol", "as_of"], keep="last"
    )
    after = len(df)
    if after < before:
        log.info(f"De-duplicated fundamentals: {before} -> {after}")
    return df
```

### 3. Updated Callers

Modified `fetch_fundamentals_for_universe()` to use the new function:

```python
def fetch_fundamentals_for_universe(batch_size: int = 50) -> pd.DataFrame:
    # ... fetch data ...
    if not df.empty:
        # Use proper ON CONFLICT upsert instead of simple append
        _upsert_fundamentals(df)
    return df
```

Also updated `data/institutional_ingest.py`:

```python
def validate_and_ingest_fundamentals(df: pd.DataFrame, source: str = "polygon") -> bool:
    try:
        # Perform ingestion using proper ON CONFLICT upsert
        from data.fundamentals import _upsert_fundamentals
        _upsert_fundamentals(df)
        # ... rest of function
```

## Files Modified

1. **`data/fundamentals.py`**
   - Added `_dedupe_fundamentals()` function
   - Added `_upsert_fundamentals()` function  
   - Updated `fetch_fundamentals_for_universe()` to use new upsert
   - Added import: `from sqlalchemy.dialects.postgresql import insert as pg_insert`
   - Removed unused import: `upsert_dataframe`

2. **`data/institutional_ingest.py`**
   - Updated `validate_and_ingest_fundamentals()` to use `_upsert_fundamentals()`

## Tests Created

### Unit Tests (`test_fundamentals_upsert_fix.py`)
- Test deduplication of duplicate (symbol, as_of) pairs
- Test handling of within-batch duplicates
- Test empty dataframe handling
- All tests pass ✓

### Integration Tests (`test_fundamentals_integration.py`)
- Test duplicate key violation scenario (insert existing record)
- Test mixed batch (duplicates + new data)
- Successfully validates the fix ✓

### Manual Validation Demo (`demo_fundamentals_fix.py`)
- Interactive demonstration of the fix
- Shows before/after data states
- Confirms records are updated instead of failing ✓

## Validation Results

### Demo Output
```
1. Creating database schema...
   ✓ Schema created

2. Inserting initial fundamentals data...
   ✓ Inserted 2 records

3. Querying initial data from database...
symbol      as_of  debt_to_equity  return_on_assets
     A 2025-08-29             1.5              0.10
  AAPL 2025-08-29             1.3              0.15

4. Attempting to insert duplicate (symbol, as_of) pair...
   ✓ Successfully upserted 1 records (no error!)

5. Querying data after upsert (should show updated values)...
symbol      as_of  debt_to_equity  return_on_assets
     A 2025-08-29             1.6              0.12  # ← Updated!
  AAPL 2025-08-29             1.3              0.15
```

## Key Benefits

1. **Prevents duplicate key violations**: Uses PostgreSQL ON CONFLICT to handle existing records
2. **Updates stale data**: When conflicts occur, existing records are updated with new values
3. **Handles within-batch duplicates**: Deduplication prevents multiple rows with same keys in a single batch
4. **Maintains consistency**: Follows the same pattern as `daily_bars` ingestion in `data/ingest.py`
5. **Chunked processing**: Uses 1000-row chunks to avoid PostgreSQL parameter limits (65535)
6. **Proper logging**: Logs deduplication events and successful upserts

## Pattern Consistency

This implementation mirrors the proven `_upsert_daily_bars()` pattern in `data/ingest.py`:

```python
# Both functions follow the same structure:
1. Deduplicate the dataframe
2. Filter to valid columns
3. Drop null primary key values
4. Use pg_insert() with ON CONFLICT DO UPDATE
5. Process in chunks
6. Log results
```

## Production Readiness

- ✅ Prevents the reported error
- ✅ Handles edge cases (nulls, duplicates, empty data)
- ✅ Tested with unit and integration tests
- ✅ Follows established patterns in the codebase
- ✅ No breaking changes to existing functionality
- ✅ Proper error handling and logging

## Expected Outcome

The EOD pipeline will now:
1. Successfully ingest fundamentals data even when records already exist
2. Update existing records with new data when conflicts occur
3. Handle within-batch duplicates automatically
4. Continue without UniqueViolation errors

The error "duplicate key value violates unique constraint 'fundamentals_pkey'" will no longer occur.
