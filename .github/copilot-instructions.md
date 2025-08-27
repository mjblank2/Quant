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

**NEVER CANCEL builds or installations. Set timeouts to 120+ minutes.**

```bash
# Install dependencies (takes ~1-2 minutes)
pip install -r requirements.txt
pip install -r requirements.extra.txt

# Install additional dev tools for linting
pip install flake8 pytest

# Verify installation
python -c "import streamlit, sqlalchemy, xgboost, cvxpy; print('Core dependencies OK')"
```

**Expected timing**: Dependency installation takes 1-2 minutes. NEVER cancel during package downloads.

## Database Setup and Migrations

**Database is REQUIRED for most functionality.**

```bash
# Run database migrations (requires DATABASE_URL)
alembic upgrade head

# Alternative: use scripts for setup
python scripts/data_infra_cli.py migrate
```

**Note**: Most functionality fails without DATABASE_URL environment variable. Set up PostgreSQL/TimescaleDB for full functionality.

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
# Build container (takes 5-10 minutes for full build)
docker build -t smallcap-quant:latest .

# Run web service
docker run -e DATABASE_URL="..." -e POLYGON_API_KEY="..." -p 8501:8501 smallcap-quant:latest

# Run worker mode
docker run -e SERVICE=worker -e DATABASE_URL="..." smallcap-quant:latest

# Run cron/pipeline mode
docker run -e SERVICE=cron -e DATABASE_URL="..." smallcap-quant:latest
```

**Expected timing**: Docker build takes 5-10 minutes. NEVER CANCEL. Set timeout to 20+ minutes.

## Testing and Validation

### Unit Tests
```bash
# Run individual test files (work without DB)
python test_alpha_factory.py          # ~1-2 seconds
python test_phase4_optimization.py    # ~1 second

# Run pytest suite (most tests require DATABASE_URL)
python -m pytest test_phase2_infrastructure.py -v  # requires DB
python -m pytest test_integration_phase2.py -v     # requires DB
```

### Code Quality
```bash
# Lint code (always run before committing)
flake8 app.py config.py --count
flake8 --max-line-length=160 .  # full project lint

# Expected: Many style violations exist - document but continue
```

**Expected timing**: Individual tests run in 1-2 seconds. Full pytest suite takes 10-30 seconds.

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
2. Run `alembic upgrade head`
3. Start Streamlit: `streamlit run app.py`
4. Verify the web interface loads at http://localhost:8501
5. Check that sidebar navigation works

### Scenario 2: Core Testing Without Database
1. Run `python test_alpha_factory.py` - should complete in ~2 seconds
2. Run `python test_phase4_optimization.py` - should show "All Phase 4 smoke tests passed!"
3. Verify no import errors in core modules

### Scenario 3: Docker Validation
1. Build: `docker build -t test-quant .`
2. Test startup: `docker run --rm -e DATABASE_URL="sqlite:///test.db" -p 8501:8501 test-quant`
3. Verify container starts without crashes

## Timing Expectations and Warnings

**NEVER CANCEL these operations:**

- **Dependency installation**: 1-2 minutes. Timeout: 180 seconds minimum.
- **Docker build**: 5-10 minutes for full build. Timeout: 1200+ seconds.
- **Database migrations**: 10-30 seconds. Timeout: 120 seconds.
- **Model training** (with data): 5-30 minutes depending on dataset. Timeout: 3600+ seconds.
- **Full pipeline execution**: 10-60 minutes depending on universe size. Timeout: 7200+ seconds.

## Troubleshooting

### Common Issues
1. **"DATABASE_URL environment variable is required"**
   - Set DATABASE_URL before running any data operations
   - Use SQLite for testing: `export DATABASE_URL="sqlite:///test.db"`

2. **Import errors for modules requiring database**
   - Most core modules require DATABASE_URL to import
   - Use test files that work without DB for validation

3. **API quota exceeded**
   - Check Polygon/Alpaca API quotas
   - Reduce universe size in testing

### What Works Without Database
- `test_alpha_factory.py` - Feature and validation testing
- `test_phase4_optimization.py` - Portfolio optimization tests
- Linting with flake8
- Docker build process
- Basic import testing of some modules

### What Requires Database
- Main Streamlit application (`app.py`)
- Data ingestion and pipeline (`run_pipeline.py`)
- Most integration tests
- Model training and backtesting
- Trade generation

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
2. **Always** run `alembic upgrade head` after DB changes
3. **Always** run linting before commits: `flake8 --max-line-length=160 .`
4. **Always** test changes with scenarios above
5. **Always** use appropriate timeouts for long-running operations
6. **Never** cancel builds, installations, or long-running data operations

## Production Notes

- Uses PostgreSQL with TimescaleDB extension for time-series data
- Integrates with Polygon.io and Alpaca Markets APIs
- Supports institutional execution algorithms (VWAP, TWAP)
- Includes advanced portfolio optimization with CVXPY
- Features walk-forward backtesting and validation
- Includes Docker deployment for Render/cloud platforms

> ⚠️ Trading is risky; always paper trade first. Ensure API entitlements and quotas are appropriate for your use case.