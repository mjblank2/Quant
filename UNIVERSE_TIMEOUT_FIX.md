# Universe Rebuild Timeout Fix

## Problem
The universe rebuild process was timing out after approximately 27 minutes during Polygon API pagination. The logs showed:

```
2025-09-14T13:06:24.814355132Z {"timestamp": "2025-09-14 13:06:24,811", "level": "ERROR", "logger": "data.universe", "message": "Error while fetching small cap tickers: HTTPSConnectionPool(host='api.polygon.io', port=443): Read timed out. (read timeout=30)", "module": "universe", "function": "_list_small_cap_symbols", "line": 145}
```

## Root Cause
1. **Individual API calls**: Each ticker required a separate API call to get market cap information
2. **No retry logic**: Used basic `requests.get()` without retry logic for network failures
3. **Short timeouts**: 30-second timeout was insufficient for large paginated responses
4. **No resilience**: Any network hiccup would fail the entire operation

## Solution
Replaced direct HTTP calls with the existing robust HTTP utility (`utils_http.py`) that provides:

### 1. Retry Logic with Exponential Backoff
- 3 retry attempts with exponential backoff
- Handles transient network errors gracefully
- Proper handling of rate limits (429 status codes)

### 2. Improved Timeouts
- Increased timeout from 30s to 60s for large API responses
- Better suited for paginated requests with many results

### 3. Structured Logging and Monitoring
- Comprehensive logging of HTTP calls with request IDs
- Response time and byte size tracking
- Better error visibility and debugging

### 4. Graceful Error Handling
- Returns empty dict on failures instead of crashing
- Continues processing remaining tickers if one fails
- Progress logging every 50 symbols for long-running operations

## Changes Made

### data/universe.py
1. **Added robust HTTP utility integration**:
   ```python
   from utils_http import get_json
   ```

2. **Created wrapper function**:
   ```python
   def _robust_get_json(url: str, params: Dict[str, Any] = None, timeout: float = 60.0) -> Dict[str, Any]:
       if HAS_UTILS_HTTP:
           return get_json(url, params=params, timeout=timeout, max_tries=3)
       else:
           # Fallback to basic requests
   ```

3. **Replaced all requests.get() calls**:
   - Main ticker listing pagination
   - Individual ticker detail requests  
   - API connection tests

4. **Added progress monitoring**:
   - Periodic logging every 50 symbols processed
   - Debug logging for each symbol added with market cap
   - Summary logging of total symbols found

### test_universe_timeout_fix.py
- Comprehensive test suite validating all changes
- Mocked HTTP calls to test retry logic
- Verified API key validation still works
- Confirmed graceful error handling

## Benefits
1. **Resilience**: Network timeouts and transient errors are automatically retried
2. **Observability**: Better logging and monitoring of the universe rebuild process
3. **Performance**: Larger timeouts and retry logic reduce failed operations
4. **Maintainability**: Uses existing battle-tested HTTP utility instead of custom logic

## Backward Compatibility
- Graceful fallback to basic requests if utils_http is not available
- All existing API key validation and error handling preserved
- No changes to function signatures or return values

## Testing
- ✅ All new tests pass (6/6)
- ✅ No regression in existing functionality  
- ✅ Code quality validated with flake8
- ✅ Module imports and loads correctly

This fix should resolve the timeout issues seen in production cron jobs by providing proper retry logic and better timeout handling for the long-running universe rebuild process.