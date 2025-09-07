# Pipeline Robustness Improvements - User Guide

This document describes the comprehensive robustness improvements implemented in the quantitative trading pipeline.

## Overview

The pipeline now includes multiple layers of protection against common failure modes:

- **Version Guard**: Prevents execution with incompatible schema versions
- **Market Calendar**: Handles non-trading days gracefully
- **Warning Deduplication**: Reduces log spam while preserving important alerts
- **Enhanced Trade Generation**: Multiple validation phases with abort mechanisms
- **Background Tasks**: Robust task management with timeout handling

## Quick Start

The enhanced pipeline can be used exactly like the original, but with additional safety checks:

```python
from run_pipeline import main

# Run the enhanced pipeline
success = main(sync_broker=False)
```

The pipeline will now automatically:
- Check schema version compatibility
- Validate market calendar (skip on weekends/holidays)
- Use enhanced trade generation with validation
- Apply warning deduplication to reduce log noise

## Feature Details

### 1. Version Guard

Ensures code and database schema compatibility:

```python
from version_guard import check_schema_version, update_schema_version

# Check compatibility
is_compatible, message = check_schema_version()
if not is_compatible:
    print(f"Version mismatch: {message}")

# Update schema version after migrations
update_schema_version("v17.2.0")
```

**Configuration**: Modify `CURRENT_SCHEMA_VERSION` and `MIN_COMPATIBLE_SCHEMA_VERSION` in `version_guard.py`

### 2. Market Calendar

Handles market holidays and trading schedules:

```python
from market_calendar import is_market_day, should_run_pipeline, get_market_calendar_info

# Check if today is a trading day
if is_market_day(date.today()):
    print("Market is open")

# Get comprehensive calendar info
info = get_market_calendar_info()
print(f"Next trading day: {info['next_market_day']}")

# Check if pipeline should run
should_run, reason = should_run_pipeline()
```

**Configuration**: Update holiday lists in `US_MARKET_HOLIDAYS_2024` and `US_MARKET_HOLIDAYS_2025`

### 3. Warning Deduplication

Intelligent rate limiting for warnings:

```python
from warning_dedup import warn_once, warn_adj_close_missing, get_warning_stats
import logging

logger = logging.getLogger(__name__)

# Rate-limited warning (will only log once per time period)
warn_once(logger, "unique_warning_key", "This warning won't spam", "default")

# Special handling for adj_close warnings
warn_adj_close_missing(logger, "table_name")

# Get statistics
stats = get_warning_stats()
print(f"Warning counts: {stats}")
```

**Rate Limits**:
- `adj_close`: 5 minutes
- `cardinality`: 1 minute  
- `default`: 30 seconds
- `critical`: 5 seconds

### 4. Enhanced Trade Generation

Multi-phase validation with abort mechanisms:

```python
from enhanced_trade_generation import enhanced_generate_today_trades, get_trade_generation_health

# Enhanced trade generation with validation
trades_df = enhanced_generate_today_trades()

# Get health status
health = get_trade_generation_health()
print(f"Trade generation status: {health['overall_status']}")
```

**Validation Phases**:
1. Trading conditions (market day, data freshness)
2. Prediction quality (variance, extreme values)
3. Portfolio construction (exposure limits, concentration)
4. Trade execution (price validation, quantity checks)

### 5. Background Task Management

Robust task execution with monitoring:

```python
from background_tasks import submit_background_task, get_task_status, submit_pipeline_task

# Submit a custom background task
task_info = submit_background_task(
    "my_task", 
    "My Long Running Task", 
    my_function, 
    timeout_seconds=3600,
    arg1="value1", 
    arg2="value2"
)

# Monitor progress
status = get_task_status("my_task")
print(f"Task status: {status.status}")
print(f"Progress: {status.progress}%")

# Submit common pipeline tasks
task = submit_pipeline_task("data_ingest", days=7)
```

**Common Pipeline Tasks**:
- `universe_rebuild`: Rebuild trading universe
- `data_ingest`: Ingest market data
- `feature_build`: Build features
- `model_train`: Train ML models
- `trade_generation`: Generate trades
- `full_pipeline`: Run complete pipeline

## Configuration

### Environment Variables

The robustness features respect existing configuration:

```bash
# Database (required)
export DATABASE_URL="postgresql+psycopg://user:pass@host:5432/db"

# Market data providers
export POLYGON_API_KEY="your_key"
export APCA_API_KEY_ID="your_alpaca_key"
export APCA_API_SECRET_KEY="your_alpaca_secret"

# Task queue (for background tasks)
export REDIS_URL="redis://localhost:6379/0"
```

### Logging

Enable structured logging for better observability:

```bash
export ENABLE_STRUCTURED_LOGGING=1
export LOG_LEVEL="INFO"
```

## Error Handling

The enhanced pipeline provides better error messages and recovery:

### Version Mismatch
```
❌ Version guard failed: Schema version mismatch: database has v17.0.0, but code requires >= v17.1.0
```
**Solution**: Run database migrations or update compatibility settings

### Non-Trading Day
```
📅 Pipeline skipped: 2025-01-01 is a market holiday (no market data)
```
**Solution**: Normal behavior, pipeline will resume on next trading day

### Trade Generation Abort
```
🛑 Trade generation deliberately aborted: Too many extreme predictions (>50.0%): 15
```
**Solution**: Investigate model predictions, check for data quality issues

### Task Timeout
```
⚠️ Task slow_task timed out after 3600 seconds
```
**Solution**: Increase timeout or optimize the task implementation

## Monitoring

### Health Checks

```python
from enhanced_trade_generation import get_trade_generation_health
from background_tasks import get_all_tasks
from warning_dedup import get_warning_stats

# Trading system health
health = get_trade_generation_health()
print(f"Overall status: {health['overall_status']}")

# Background task status
tasks = get_all_tasks()
active_tasks = [t for t in tasks.values() if t.status in ['pending', 'running']]
print(f"Active tasks: {len(active_tasks)}")

# Warning statistics
stats = get_warning_stats()
print(f"Warning patterns: {stats}")
```

### Structured Logging

With structured logging enabled, logs are JSON formatted for easy parsing:

```json
{
  "timestamp": "2025-09-07 21:16:55,803",
  "level": "INFO", 
  "logger": "run_pipeline",
  "message": "🚀 Starting enhanced pipeline (sync_to_broker=False)",
  "module": "run_pipeline",
  "function": "main",
  "line": 112
}
```

## Testing

Run the robustness test suite:

```bash
# Test all robustness features
python test_pipeline_robustness.py

# Run the interactive demo
python demo_robustness_improvements.py
```

## Migration from Original Pipeline

The enhanced pipeline is backward compatible. To migrate:

1. **No code changes required** - existing calls to `main()` work unchanged
2. **Database migrations** - run `alembic upgrade heads` for version tracking
3. **Optional configuration** - set up Redis for background tasks
4. **Monitoring** - review health checks and structured logging

## Best Practices

1. **Use background tasks** for long-running operations
2. **Monitor health status** regularly 
3. **Review warning statistics** to identify recurring issues
4. **Test on non-trading days** to validate calendar handling
5. **Use structured logging** in production for better observability

## Troubleshooting

Common issues and solutions:

| Issue | Symptom | Solution |
|-------|---------|----------|
| Schema version mismatch | Pipeline fails at startup | Run migrations or update version compatibility |
| Weekend execution | Pipeline skips with market calendar message | Normal behavior, resume on trading days |
| Log spam | Repeated identical warnings | Warning deduplication is working correctly |
| Task timeouts | Background tasks marked as timed out | Increase timeout or optimize task logic |
| Trade generation aborts | No trades generated with abort message | Check market conditions and prediction quality |

For additional support, review the demo script output and test results for examples of expected behavior.