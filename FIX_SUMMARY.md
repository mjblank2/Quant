# Fix Summary: chunk_size Parameter Issue

## Problem Statement
The pipeline was failing with the following error:
```
TypeError: upsert_dataframe() got an unexpected keyword argument 'chunk_size'
```

This error occurred in `models/features.py` at line 362 when calling:
```python
upsert_dataframe(feats, Feature, ['symbol', 'ts'], chunk_size=200)
```

## Root Cause
The `upsert_dataframe()` function in `db.py` did not have a `chunk_size` parameter in its signature, but multiple files were calling it with this parameter:
- `models/features.py` (line 362): `chunk_size=200`
- `features/build_features.py` (line 340): `chunk_size=200`
- `features/store.py` (line 63): `chunk_size=200`
- Multiple test files with various chunk_size values

## Solution
Updated the `upsert_dataframe()` function signature in `db.py` to accept an optional `chunk_size` parameter:

### Before:
```python
def upsert_dataframe(
    df: pd.DataFrame,
    table: Any,
    conflict_cols: Optional[list[str]] = None,
    update_cols: Optional[list[str]] = None,
) -> None:
    ...
    filtered_df.to_sql(
        table_name,
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,  # hardcoded
    )
```

### After:
```python
def upsert_dataframe(
    df: pd.DataFrame,
    table: Any,
    conflict_cols: Optional[list[str]] = None,
    update_cols: Optional[list[str]] = None,
    chunk_size: Optional[int] = None,
) -> None:
    """
    ...
    chunk_size : Optional[int]
        Number of rows to insert per batch. Defaults to 1000 if not specified.
    """
    ...
    chunksize = chunk_size if chunk_size is not None else 1000
    filtered_df.to_sql(
        table_name,
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=chunksize,  # uses parameter or default
    )
```

## Changes Made
1. Added `chunk_size: Optional[int] = None` parameter to function signature
2. Added parameter documentation to docstring
3. Modified the function to use the provided `chunk_size` or default to 1000
4. Fixed flake8 linting issues (whitespace)

## Testing
All tests pass successfully:
- ✅ Phase 2 infrastructure tests (8/8 passed)
- ✅ Integration tests (4/4 passed)
- ✅ Comprehensive chunk_size fix test (all scenarios passed)
- ✅ Flake8 linting (0 errors)
- ✅ All affected modules import successfully
- ✅ Backward compatible (works without the parameter)
- ✅ Works with various chunk sizes (50, 100, 200, 500, 1000)

## Verification
The exact error scenario from the problem statement has been resolved:
```bash
$ python -c "
import os
os.environ['DATABASE_URL'] = 'sqlite:///test.db'
import pandas as pd
import db
db.create_tables()
test_df = pd.DataFrame({
    'symbol': ['TEST1'],
    'ts': pd.Timestamp('2025-10-03'),
    'ret_1d': [0.01]
})
db.upsert_dataframe(test_df, db.Feature, ['symbol', 'ts'], chunk_size=200)
print('✅ No TypeError - fix is working!')
"
✅ No TypeError - fix is working!
```

## Files Modified
- `db.py`: Updated `upsert_dataframe()` function signature

## Files Added
- `test_chunk_size_fix.py`: Comprehensive test for the fix

## Impact
- **Minimal change**: Only 1 file modified (db.py)
- **Backward compatible**: Existing code without chunk_size parameter still works
- **Solves the issue**: All affected files can now use the chunk_size parameter
- **No breaking changes**: All existing tests pass
