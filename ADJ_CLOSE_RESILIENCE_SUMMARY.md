# adj_close Resilience Implementation Summary

This refactoring successfully removes the hard dependency on the `adj_close` column throughout the entire codebase, making the system resilient to databases that lack this column.

## What Changed

### Core Infrastructure
- **`utils/price_utils.py`**: Enhanced with `select_price_as()` helper function and improved documentation
- **Centralized Logic**: All price queries now use dynamic `price_expr()` instead of hardcoded `COALESCE(adj_close, close)`

### Modules Refactored (14 files)
- `features/build_features.py` - Removed `_has_adj_close()` duplication
- `features/store.py` - Simplified try/except to single dynamic query  
- `risk/risk_model.py` - Dynamic price expressions
- `risk/factor_model.py` - Dynamic price expressions
- `portfolio/mvo.py` - Dynamic price expressions
- `portfolio/build.py` - Dynamic price expressions
- `portfolio/heuristic.py` - Dynamic price expressions
- `portfolio/optimizer.py` - Dynamic price expressions
- `trading/generate_trades.py` - Dynamic price expressions
- `hedges/overlays.py` - Dynamic price expressions
- `app.py` - Streamlit price chart queries
- `scripts/backfill_forward_returns.py` - Batch processing scripts

### Documentation & Verification
- **`README_v17.md`**: Updated section with resilience guarantees
- **`scripts/verify_adj_close_resilience.py`**: Verification script for any environment
- **`test_adj_close_resilience.py`**: Comprehensive test suite

## Benefits Achieved

### ✅ Resilience
- **No more crashes**: `psycopg.errors.UndefinedColumn` eliminated
- **Hot-fix ready**: Works immediately without schema changes
- **Forward compatible**: Automatically uses adjusted prices when column added

### ✅ Maintainability  
- **Single source of truth**: All price logic centralized in `utils/price_utils.py`
- **No duplication**: Removed multiple `_has_adj_close()` implementations
- **Consistent behavior**: All modules use same detection and fallback logic

### ✅ Compatibility
- **DataFrame schemas preserved**: Output columns maintain expected names (`adj_close` alias)
- **No breaking changes**: Existing code depending on column names continues to work
- **Migration-free**: Existing databases work without modification

## Testing Results

- ✅ **Column detection**: Correctly identifies presence/absence of `adj_close`
- ✅ **SQL generation**: Produces valid queries for both scenarios  
- ✅ **Fallback behavior**: Uses `close` when `adj_close` missing
- ✅ **Preference behavior**: Uses `COALESCE(adj_close, close)` when available
- ✅ **Logging**: Single info message indicates active mode
- ✅ **Import integrity**: All refactored modules import successfully

## Production Impact

**Before**: EOD pipeline would crash with `UndefinedColumn` error if `adj_close` missing
**After**: Pipeline runs successfully regardless of column presence, with transparent price handling

This implementation fulfills all requirements from the problem statement and provides a robust foundation for price handling across the quantitative trading system.