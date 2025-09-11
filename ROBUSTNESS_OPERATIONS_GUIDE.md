# Data Robustness & Modeling Resiliency - Operational Guide

This document describes the bootstrap and steady-state operational sequences for the enhanced data robustness and modeling resiliency features implemented in the quantitative trading system.

## Overview

The system now includes enhanced data robustness features that provide:
- Adaptive historical data backfill with resume capabilities
- Target coverage monitoring and automatic fallback logic
- Symbol breadth validation with configurable thresholds
- Comprehensive data validation and monitoring tools

## Bootstrap Sequence (Initial Setup)

### 1. Environment Configuration

Set required environment variables:

```bash
# Core database and API configuration
export DATABASE_URL="postgresql+psycopg://user:pass@host:5432/dbname"
export POLYGON_API_KEY="your_polygon_key"
export APCA_API_KEY_ID="your_alpaca_key"
export APCA_API_SECRET_KEY="your_alpaca_secret"

# Data robustness configuration
export BACKFILL_TARGET_DAYS="730"          # 2 years of historical data
export MIN_TARGET_COVERAGE="0.6"           # 60% minimum target coverage
export FEATURE_BREADTH_MIN_SYMBOLS="100"   # Critical symbol threshold
export FEATURE_BREADTH_WARN_SYMBOLS="200"  # Warning symbol threshold

# Backfill performance tuning
export BACKFILL_MIN_STEP="15"              # Minimum backfill step size
export BACKFILL_MAX_STEP="90"              # Maximum backfill step size  
export BACKFILL_SLOW_THRESHOLD="240"       # Slow operation threshold (seconds)
```

### 2. Database Schema Setup

```bash
# Run database migrations to create required tables
alembic upgrade heads

# Ensure universe table has required columns
python scripts/ensure_universe_columns.py
```

### 3. Initial Data Population

Execute the bootstrap sequence in order:

```bash
# Step 1: Build initial universe
python -c "from data.universe import rebuild_universe; print(f'Universe built: {len(rebuild_universe())} symbols')"

# Step 2: Historical data backfill (adaptive with resume)
python scripts/historical_backfill.py

# Step 3: Populate fundamentals data
python scripts/upsert_fundamentals.py

# Step 4: Compute ADV fallback values
python scripts/recompute_adv_fallback.py

# Step 5: Generate features
python -m models.features

# Step 6: Initial model training (with target fallback)
python -c "from models.ml import train_and_predict_all_models; train_and_predict_all_models()"
```

### 4. Validation After Bootstrap

```bash
# Check target coverage health
python scripts/debug_target_coverage.py

# Validate symbol breadth
python scripts/feature_breadth_check.py

# Run data validation pipeline
python scripts/data_infra_cli.py validate --full
```

## Steady-State Operations

### Daily EOD Pipeline

The enhanced pipeline includes automatic robustness checks:

```bash
# Standard EOD pipeline (now includes robustness features)
python run_pipeline.py
```

The pipeline automatically:
1. Validates data freshness and completeness
2. Monitors target coverage and applies fallback if needed
3. Checks symbol breadth against thresholds
4. Logs robustness metrics for monitoring

### Weekly Health Checks

Run comprehensive validation weekly:

```bash
# Weekly validation script
#!/bin/bash
set -e

echo "=== Weekly Data Health Check ==="

# Check target coverage over extended period
python scripts/debug_target_coverage.py

# Validate symbol breadth
python scripts/feature_breadth_check.py

# Check data infrastructure health
python scripts/data_infra_cli.py health-check

# Validate TimescaleDB status (if enabled)
python scripts/data_infra_cli.py timescale --check

echo "=== Health check complete ==="
```

### Monthly Maintenance

Perform monthly maintenance tasks:

```bash
# Monthly maintenance script
#!/bin/bash
set -e

echo "=== Monthly Maintenance ==="

# Refresh fundamentals data
python scripts/upsert_fundamentals.py

# Recompute ADV with latest data
python scripts/recompute_adv_fallback.py

# Extended data validation
python scripts/data_infra_cli.py validate --full

# Clean up old ingestion progress records (optional)
# python -c "
# from db import engine
# from sqlalchemy import text
# with engine.begin() as c:
#     c.execute(text('DELETE FROM ingestion_progress WHERE run_ts < NOW() - INTERVAL \'90 days\''))
# print('Cleaned old progress records')
# "

echo "=== Monthly maintenance complete ==="
```

## Operational Monitoring

### Key Metrics to Monitor

1. **Target Coverage Metrics**:
   - fwd_ret coverage percentage
   - fwd_ret_resid coverage percentage
   - Fallback activation frequency

2. **Symbol Breadth Metrics**:
   - Total symbols with features
   - Symbol count vs. thresholds
   - Coverage rate within features

3. **Data Freshness**:
   - Latest feature date
   - Data ingestion lag
   - Missing data gaps

### Alert Thresholds

Configure monitoring alerts for:

- Target coverage below 60% (critical)
- Symbol count below 100 (critical)
- Symbol count below 200 (warning)
- Feature coverage rate below 90% (warning)
- Data lag exceeding 2 business days (warning)

### Log Monitoring

Monitor logs for key indicators:

```bash
# Key log patterns to watch
grep "falling back to" logs/app.log        # Target fallback activations
grep "Symbol count.*below.*threshold" logs/app.log  # Breadth warnings
grep "Coverage rate.*low" logs/app.log     # Coverage warnings
grep "Failed to" logs/app.log              # General failures
```

## Emergency Procedures

### Data Quality Issues

If data quality issues are detected:

1. **Immediate Assessment**:
   ```bash
   python scripts/debug_target_coverage.py
   python scripts/feature_breadth_check.py
   ```

2. **Targeted Remediation**:
   ```bash
   # For specific symbols/dates (when implemented)
   python scripts/ingest_date_range.py --symbol AAPL --start-date 2024-01-01 --end-date 2024-01-31
   
   # For broader issues, re-run backfill
   python scripts/historical_backfill.py
   ```

3. **Validation**:
   ```bash
   python scripts/data_infra_cli.py validate --full
   ```

### Model Performance Degradation

If models show poor performance:

1. **Check Target Coverage**:
   ```bash
   python scripts/debug_target_coverage.py
   ```

2. **Force Model Retraining**:
   ```bash
   python -c "from models.ml import train_and_predict_all_models; train_and_predict_all_models(window_years=5)"
   ```

3. **Validate Feature Completeness**:
   ```bash
   python scripts/feature_breadth_check.py
   python -m models.features  # Regenerate if needed
   ```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKFILL_TARGET_DAYS` | 420 | Target days for historical backfill |
| `MIN_TARGET_COVERAGE` | 0.6 | Minimum target coverage before fallback |
| `FEATURE_BREADTH_MIN_SYMBOLS` | 100 | Critical symbol count threshold |
| `FEATURE_BREADTH_WARN_SYMBOLS` | 200 | Warning symbol count threshold |
| `BACKFILL_MIN_STEP` | 15 | Minimum backfill step size |
| `BACKFILL_MAX_STEP` | 90 | Maximum backfill step size |
| `BACKFILL_SLOW_THRESHOLD` | 240 | Slow operation threshold (seconds) |

### Script Usage Examples

```bash
# Debug target coverage for last 60 days
DAYS_BACK=60 python scripts/debug_target_coverage.py

# Check breadth with custom thresholds
FEATURE_BREADTH_MIN_SYMBOLS=150 python scripts/feature_breadth_check.py

# Backfill with custom target
BACKFILL_TARGET_DAYS=1000 python scripts/historical_backfill.py

# Coverage fallback with higher threshold
MIN_TARGET_COVERAGE=0.8 python -c "from models.ml import train_and_predict_all_models; train_and_predict_all_models()"
```

## Integration with Existing Systems

### Streamlit UI Integration

The robustness features integrate with the existing Streamlit interface:

- Task queue support for long-running backfill operations
- Real-time monitoring of robustness metrics
- Interactive validation reports
- Alert dashboards for threshold violations

### Celery Task Queue

Robustness operations can be dispatched asynchronously:

```python
# Example task queue integration
from tasks import dispatch_task

# Async backfill
dispatch_task("historical_backfill", {})

# Async validation
dispatch_task("run_validation", {"type": "full"})

# Async maintenance
dispatch_task("monthly_maintenance", {})
```

### Docker Deployment

The enhanced system works seamlessly in Docker:

```dockerfile
# Environment variables in Docker
ENV BACKFILL_TARGET_DAYS=730
ENV MIN_TARGET_COVERAGE=0.6
ENV FEATURE_BREADTH_MIN_SYMBOLS=100

# Health check integration
HEALTHCHECK --interval=30m --timeout=10s --start-period=5m --retries=3 \
  CMD python scripts/feature_breadth_check.py || exit 1
```

## Performance Considerations

### Backfill Performance

- Start with smaller `BACKFILL_MAX_STEP` values for stability
- Monitor API rate limits and adjust `BACKFILL_SLEEP_BETWEEN_SEC`
- Use resume capabilities for long backfill operations

### Memory Usage

- Large target coverage analysis may require significant memory
- Consider reducing analysis window for systems with limited RAM
- Monitor memory usage during batch operations

### Database Load

- Robustness checks query large datasets
- Consider running during off-peak hours
- Use read replicas for validation queries if available

## Troubleshooting

### Common Issues

1. **High Memory Usage During Coverage Analysis**:
   - Reduce analysis window in debug_target_coverage.py
   - Run validation in smaller batches

2. **Slow Backfill Performance**:
   - Increase `BACKFILL_SLOW_THRESHOLD`
   - Reduce `BACKFILL_MAX_STEP`
   - Check API rate limits

3. **Frequent Target Fallbacks**:
   - Investigate data quality issues
   - Adjust `MIN_TARGET_COVERAGE` threshold
   - Check fundamental data completeness

4. **Symbol Breadth Warnings**:
   - Verify universe definition
   - Check data ingestion completeness
   - Review API quotas and limits

### Log Analysis

Monitor application logs for robustness-related messages:

```bash
# Coverage analysis logs
grep "Target coverage analysis" logs/app.log

# Fallback activations
grep "falling back to" logs/app.log

# Breadth validation
grep "Symbol breadth" logs/app.log

# Backfill progress
grep "historical_backfill" logs/app.log
```

This operational guide provides comprehensive procedures for both bootstrap initialization and ongoing steady-state operations of the enhanced data robustness and modeling resiliency features.