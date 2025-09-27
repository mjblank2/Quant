# Cronjob Troubleshooting Guide

This guide helps diagnose and fix cronjob failures that result in "Exited with status 1" errors.

## Enhanced Error Handling

The pipeline has been enhanced with comprehensive error handling and diagnostics:

### ✅ Improved Logging
- Clear status indicators with emojis (✅, ❌, ⚠️, 🚀, etc.)
- Timestamp logging for execution tracking
- Exit code capture and reporting
- Full traceback logging for debugging

### ✅ Dependency Management
- Automatic dependency checking before execution
- Graceful degradation for optional components (xgboost, data validation)
- Safe module importing with individual error handling
- Clear indication of which components are available/unavailable

### ✅ Database Handling
- Automatic database migration attempt
- Fallback to basic schema creation if migrations fail
- Database connection verification before proceeding
- Environment variable validation

## Common Issues and Solutions

### 1. Missing DATABASE_URL
**Error**: `❌ ERROR: DATABASE_URL environment variable is required`
**Solution**: Ensure DATABASE_URL is configured in your environment

### 2. Missing Dependencies
**Error**: `❌ Dependency check failed: Missing required dependencies: xgboost`
**Solution**: The pipeline will gracefully skip optional components. Core functionality continues.

### 3. Database Schema Issues
**Error**: `❌ Database tables not found - migrations may need to be run`
**Solution**: The pipeline automatically attempts to create basic schema as fallback

### 4. Import Failures
**Error**: `❌ Module import failed`
**Solution**: Check the specific import error in logs. Pipeline handles missing optional modules gracefully.

## Reading the Logs

### Success Indicators
```
✅ Dependencies check passed: All core dependencies available
✅ Database check passed: Database connection and basic tables verified
✅ Module imports successful
✅ Bar ingestion completed
✅ Feature engineering completed
🎉 Pipeline completed successfully
```

### Warning Indicators
```
⚠️ models.ml import failed (may be due to missing xgboost)
⚠️ Skipping model training (dependencies not available)
⚠️ Data validation skipped or failed softly
```

### Error Indicators
```
❌ Dependency check failed
❌ Database migration failed
❌ Bar ingestion failed
❌ Pipeline execution failed
❌ SQLAlchemy parameter limit exceeded (f405 error)
```

## Exit Codes

- **0**: Success - all operations completed successfully
- **1**: Failure - one or more critical operations failed
- **130**: Interrupted - execution was stopped by user (SIGINT)

## Common Issues and Solutions

### curl to API returns 404 or 405 from cron

Symptoms:
- 404 Not Found when POSTing to a wrong API hostname or missing /ingest path
- 405 Method Not Allowed when POSTing to the root (/) instead of /ingest

Resolution (recommended):
- Switch cron to direct Python execution (Option A) using the provided scripts:
  - scripts/cron_universe.sh
  - scripts/cron_ingest.sh
  - scripts/cron_eod_pipeline.sh

This removes the dependency on the API/Celery path for scheduled operations. Keep curl-based triggers only for ad‑hoc/manual runs.

### Parameter Limit Errors (f405) - FIXED

**Issue**: SQL parameter limit exceeded with parameters like `%(symbol_m795)s::VARCHAR`
**Root Cause**: Hard-coded 1000 row cap in `upsert_dataframe` could exceed database parameter limits for tables with many columns
**Fix Applied**: Removed hard cap and rely on calculated `theoretical_max_rows` based on actual database limits

**Details**:
- SQLite limit: 999 parameters  
- PostgreSQL limit: 16,000 parameters
- Fix ensures batches respect `max_params / num_columns` calculation
- No longer artificially limited to 1000 rows regardless of column count

**Symptoms:**
- Error message mentioning "f405" 
- Parameters listed as 'symbol_m0', 'ts_m0', etc. up to large numbers (e.g., 'm2055')
- "Exited with status 1" from cronjob

**Root Cause:**
PostgreSQL has varying parameter limits (as low as 16,000 in some configurations) and bulk operations can exceed these limits.

**Solution:**
The system now automatically:
- Uses conservative parameter limits (10,000 max)
- Caps batch sizes at 1,000 rows maximum
- Implements retry logic with smaller batches on parameter limit errors

**Fixed in version:** This issue has been resolved in the current version.

## Debugging Steps

1. **Check Environment Variables**
   ```bash
   echo "DATABASE_URL: ${DATABASE_URL:+SET}"
   echo "POLYGON_API_KEY: ${POLYGON_API_KEY:+SET}"
   echo "ALPACA_API_KEY: ${ALPACA_API_KEY:+SET}"
   ```

2. **Test Pipeline Manually**
   ```bash
   python run_pipeline.py
   ```

3. **Check Database Connection**
   ```bash
   python -c "from config import DATABASE_URL; print('DB URL:', DATABASE_URL[:20] + '...')"
   ```

4. **Verify Module Imports**
   ```bash
   python -c "import data.ingest, models.features; print('Core modules OK')"
   ```

## Expected Behavior

### In Development Environment
- Pipeline may skip model training due to missing xgboost
- Data validation may be skipped if great-expectations not installed
- Trade generation may fail if predictions table doesn't exist
- **This is expected** - pipeline provides clear warnings

### In Production Environment
- All dependencies should be available
- Database should be properly migrated
- All phases should complete successfully
- Exit code should be 0

## Getting Help

1. **Check the logs first** - they now provide detailed information
2. **Look for specific error messages** - each failure mode has clear indicators
3. **Check this troubleshooting guide** for common solutions
4. **Verify environment configuration** using the debugging steps above

The enhanced error handling ensures that when cronjobs fail, you'll have clear information about what went wrong and how to fix it.