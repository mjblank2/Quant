# Small-Cap Quantitative Trading System

**Always follow these instructions first and fallback to search or bash commands only when encountering unexpected information that does not match the info here.**

The Small-Cap Quant System is a sophisticated quantitative trading platform built with Python, featuring Polygon/Alpaca market data integration, machine learning models, portfolio optimization, and institutional-grade execution algorithms.

## Quick Start & Environment Setup

**CRITICAL**: Set up environment variables first before any operations:
```bash
# Copy environment template and configure
cp .env .env.local  # edit with your DATABASE_URL, API keys
export DATABASE_URL="postgresql+psycopg://user:pass@host:5432/dbname"
export POLYGON_API_KEY="your_polygon_key"
export ALPACA_API_KEY="your_alpaca_key"
export ALPACA_SECRET_KEY="your_alpaca_secret"
```

## Building and Dependencies

**NEVER CANCEL builds or installations. Set timeouts to 300+ seconds.**

```bash
# Install dependencies (takes ~1-2 minutes)
pip install -r requirements.txt
pip install -r requirements.extra.txt

# Install additional dev tools for linting
pip install flake8 pytest

# Verify installation
python -c "import streamlit, sqlalchemy, xgboost, cvxpy; print('Core dependencies OK')"
```

**Actual timing measured**: 
- requirements.txt: 1m 13s
- requirements.extra.txt: 1.4s  
- dev tools: 3.9s
- Total: ~1-2 minutes as documented. NEVER cancel during package downloads.

## Database Setup and Migrations

**Database is REQUIRED for most functionality. Known issue with multiple migration heads.**

```bash
# RECOMMENDED: Use alembic heads instead of head due to multiple head issue
alembic upgrade heads

# Alternative: try data_infra_cli (also affected by migration conflicts)
python scripts/data_infra_cli.py migrate

# For testing: use SQLite (may have migration conflicts)
export DATABASE_URL="sqlite:///test.db"
```

**Known Issues**: The database migration system currently has multiple head revisions that cause conflicts. This affects both `alembic upgrade head` and the data_infra_cli script. Use `alembic upgrade heads` or skip database-dependent features for validation.

## Running the Application

### Streamlit Web Application
```bash
# Main operator interface (requires DATABASE_URL)
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Alternative dashboard (may work with limited DB access)
streamlit run data_ingestion/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

### Docker Deployment
```bash
# Build container (may fail in CI environments due to SSL issues)
docker build -t smallcap-quant:latest .

# Run web service
docker run -e DATABASE_URL="..." -e POLYGON_API_KEY="..." -p 8501:8501 smallcap-quant:latest

# Run worker mode
docker run -e SERVICE=worker -e DATABASE_URL="..." smallcap-quant:latest

# Run cron/pipeline mode
docker run -e SERVICE=cron -e DATABASE_URL="..." smallcap-quant:latest
```

**Docker Build Issues**: In CI environments, Docker build may fail at ~43 seconds due to SSL certificate verification errors with PyPI. This is an environment-specific issue, not a fundamental Dockerfile problem. In normal environments, build takes 5-10 minutes. NEVER CANCEL. Set timeout to 1200+ seconds.

## Testing and Validation

### Unit Tests
```bash
# Run individual test files (work without DB)
python test_alpha_factory.py          # ~1.2 seconds
python test_phase4_optimization.py    # ~1.1 seconds

# Run pytest suite (most tests require DATABASE_URL)
python -m pytest test_phase2_infrastructure.py -v  # ~1.1 seconds, works despite DB issues
python -m pytest test_integration_phase2.py -v     # requires DB
```

### Code Quality
```bash
# Lint code (always run before committing)
flake8 app.py config.py --count
flake8 --max-line-length=160 .  # full project lint

# Expected: Many style violations exist - document but continue
```

**Actual timing measured**: Individual tests run in 1.1-1.2 seconds. Pytest suite takes 1.1 seconds. All timing is accurate as documented.

## Core Workflows

### Data Pipeline (Full EOD Process)
```bash
# Full production pipeline (requires API keys + DATABASE_URL)
python run_pipeline.py

# Individual components
python -m models.features          # Feature engineering
python -c "from validation.wfo import run_wfo; print(run_wfo())"  # Walk-forward optimization (if available)
```

### Key Modules and Entry Points
```bash
# Feature engineering
python -m models.features

# ML model training and prediction
python -m models.ml

# Trade generation
python trading/generate_trades.py

# Data ingestion
python data/ingest.py
```

## Manual Validation Scenarios

**ALWAYS test these complete scenarios after making changes:**

### Scenario 1: Basic Application Startup
1. Set DATABASE_URL environment variable
2. Run `alembic upgrade heads` (note: use 'heads' not 'head' due to migration conflicts)
3. Start Streamlit: `streamlit run app.py` (starts successfully despite DB migration issues)
4. Verify the web interface loads at http://localhost:8501
5. Check that sidebar navigation works

### Scenario 2: Core Testing Without Database
1. Run `python test_alpha_factory.py` - completes in ~1.2 seconds, shows "🎉 All tests passed!"
2. Run `python test_phase4_optimization.py` - completes in ~1.1 seconds, shows "🎉 All Phase 4 smoke tests passed!"
3. Verify no import errors in core modules: `python -c "from models import features, ml; from data import ingest; from trading import generate_trades; print('All imports successful')"`

### Scenario 3: Docker Validation
1. Build: `docker build -t test-quant .` (may fail in CI environments due to SSL issues)
2. Test startup: `docker run --rm -e DATABASE_URL="sqlite:///test.db" -p 8501:8501 test-quant` (if build succeeds)
3. Verify container starts without crashes

### Scenario 4: Verification Scripts
1. Run `python scripts/verify_render_setup.py` - should complete in ~0.06 seconds showing "🎉 All checks passed!"
2. Test entrypoint: `chmod +x scripts/entrypoint.sh && export SERVICE=web APP_MODE=streamlit && scripts/entrypoint.sh` (works with migration warnings)

## Timing Expectations and Warnings

**NEVER CANCEL these operations:**

- **Dependency installation**: 1-2 minutes (measured: requirements.txt 1m13s, extra 1.4s). Timeout: 300+ seconds.
- **Docker build**: 5-10 minutes for full build (fails in CI at 43s due to SSL issues). Timeout: 1200+ seconds.
- **Database migrations**: 10-30 seconds (may fail due to multiple head conflicts). Timeout: 120 seconds.
- **Model training** (with data): 5-30 minutes depending on dataset. Timeout: 3600+ seconds.
- **Full pipeline execution**: 10-60 minutes depending on universe size. Timeout: 7200+ seconds.
- **Individual tests**: 1-2 seconds each (measured: 1.1-1.2s). Timeout: 60 seconds.
- **Verification scripts**: Under 1 second (measured: 0.06s). Timeout: 30 seconds.

## Troubleshooting

### Common Issues
1. **"DATABASE_URL environment variable is required"**
   - Set DATABASE_URL before running any data operations
   - Use SQLite for testing: `export DATABASE_URL="sqlite:///test.db"`

2. **Database migration multiple head conflicts**
   - Use `alembic upgrade heads` instead of `alembic upgrade head`
   - Known issue: migrations have conflicting head revisions
   - Alternative: skip database-dependent features for testing

3. **Import errors for modules requiring database**
   - Most core modules require DATABASE_URL to import
   - Use test files that work without DB for validation

4. **API quota exceeded**
   - Check Polygon/Alpaca API quotas
   - Reduce universe size in testing

5. **Docker build SSL failures in CI**
   - Common in CI environments due to certificate verification
   - Not a fundamental Dockerfile issue
   - Works in normal development environments

### What Works Without Database
- `test_alpha_factory.py` - Feature and validation testing (1.2s)
- `test_phase4_optimization.py` - Portfolio optimization tests (1.1s)
- `test_phase2_infrastructure.py` - Infrastructure tests (1.1s, works despite DB issues)
- Linting with flake8 (shows expected violations)
- Docker build process (may fail in CI due to SSL)
- Basic import testing of core modules
- Streamlit application startup (loads with migration warnings)
- `scripts/verify_render_setup.py` - Deployment verification (0.06s)

### What Requires Database
- Data ingestion and pipeline (`run_pipeline.py`)
- Full integration tests without mocking
- Model training and backtesting with real data
- Trade generation with real data
- Database-dependent Streamlit features

## Key File Locations

**Configuration**: `config.py` - All environment variables and settings
**Main Entry**: `app.py` - Streamlit web application
**Pipeline**: `run_pipeline.py` - Full EOD data processing
**Database**: `db.py` - SQLAlchemy models and connections
**Models**: `models/` - ML model training and prediction
**Data**: `data/` - Ingestion, validation, and processing
**Trading**: `trading/` - Order generation and broker integration
**Tests**: `test_*.py` - Unit and integration tests
**Docker**: `Dockerfile`, `scripts/entrypoint.sh` - Containerization
**Migrations**: `alembic/` - Database schema management

## Development Workflow

1. **Always** set environment variables first
2. **Always** run `alembic upgrade heads` after DB changes (note: use 'heads' not 'head')
3. **Always** run linting before commits: `flake8 --max-line-length=160 .`
4. **Always** test changes with validation scenarios above
5. **Always** use appropriate timeouts for long-running operations (see timing section)
6. **Never** cancel builds, installations, or long-running data operations
7. **Always** test both individual test files for quick validation
8. **Always** run `python scripts/verify_render_setup.py` before deployment

## Production Notes

- Uses PostgreSQL with TimescaleDB extension for time-series data
- Integrates with Polygon.io and Alpaca Markets APIs
- Supports institutional execution algorithms (VWAP, TWAP)
- Includes advanced portfolio optimization with CVXPY
- Features walk-forward backtesting and validation
- Includes Docker deployment for Render/cloud platforms

## Validation Summary

**Last validated**: August 2025
**Validation environment**: CI/Ubuntu with Python 3.12

**Working commands** (all tested and timing verified):
- `pip install -r requirements.txt` (1m 13s)
- `pip install -r requirements.extra.txt` (1.4s)
- `python test_alpha_factory.py` (1.2s)
- `python test_phase4_optimization.py` (1.1s)
- `python -m pytest test_phase2_infrastructure.py -v` (1.1s)
- `streamlit run app.py` (starts successfully)
- `streamlit run data_ingestion/dashboard.py` (starts successfully)
- `flake8 --max-line-length=160 .` (shows expected violations)
- `python scripts/verify_render_setup.py` (0.06s, passes all checks)

**Known issues to document**:
- Database migrations have multiple head conflicts - use `alembic upgrade heads`
- Docker build fails in CI environments due to SSL certificate issues
- Most functionality works despite migration warnings

> ⚠️ Trading is risky; always paper trade first. Ensure API entitlements and quotas are appropriate for your use case.