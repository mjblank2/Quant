# Small-Cap Quantitative Trading System (v17)

**Always follow these instructions first and fall back to search or ad‑hoc commands only when something doesn’t match what’s documented here.**

This system is a small-cap quantitative trading platform in Python with:
- Market data via Polygon (PTI), Alpaca Markets, and optional Tiingo
- ML models (XGBoost, Random Forest, Ridge) with blending and regime gating
- Advanced portfolio optimization (CVXPY MVO with robust fallbacks)
- Point-in-time data governance (available_at, knowledge_date) and bi-temporal support
- TimescaleDB optional acceleration for time-series tables
- Task queue (Celery + Redis) for async pipelines
- Streamlit operator UI and a generic data dashboard


## Quick Start & Environment Setup

CRITICAL: Export environment variables before any operation.

```bash
# Copy and customize local env (example)
cp .env .env.local  # then edit with your connection string and API keys

# Database DSN (PostgreSQL recommended). The app auto-normalizes postgres:// to postgresql+psycopg://
export DATABASE_URL="postgresql+psycopg://user:pass@host:5432/dbname"

# Market data providers
export POLYGON_API_KEY="your_polygon_key"
# Prefer APCA_*; ALPACA_* is still accepted as fallback
export APCA_API_KEY_ID="your_alpaca_key"
export APCA_API_SECRET_KEY="your_alpaca_secret"
export APCA_API_BASE_URL="https://paper-api.alpaca.markets"  # paper by default
export ALPACA_DATA_FEED="sip"  # or 'iex' on some plans
export TIINGO_API_KEY="your_tiingo_key"  # optional

# Task queue
export REDIS_URL="redis://localhost:6379/0"
export CELERY_BROKER_URL="$REDIS_URL"
export CELERY_RESULT_BACKEND="$REDIS_URL"

# Feature toggles (defaults shown)
export ENABLE_STRUCTURED_LOGGING=1
export ENABLE_TIMESCALEDB=1
export ENABLE_DATA_VALIDATION=1
export USE_UNIVERSE_HISTORY=1
export REGIME_GATING=1
```


## Building and Dependencies

NEVER CANCEL builds or installations. Set timeouts >= 300 seconds.

```bash
# Base dependencies (~1–2 minutes in CI)
pip install -r requirements.txt
# Extra pin for psycopg wheels (optional on many systems)
pip install -r requirements.extra.txt

# Dev tools
pip install flake8 pytest pytest-asyncio

# Smoke import check
python - <<'PY'
import streamlit, sqlalchemy, xgboost, cvxpy, celery, fastapi, great_expectations
print('Core + optional deps OK')
PY
```

Actual timing (reference, Ubuntu CI):
- requirements.txt: ~1m 10s–1m 20s
- requirements.extra.txt: ~1–2s
- Dev tools: ~4s


## Database Setup and Migrations

A database is required for most functionality. The migration history may have multiple heads; always upgrade to heads.

```bash
# Preferred (handles multi-head):
alembic upgrade heads

# If Alembic CLI is confused by PYTHONPATH shadowing, run with a clean path:
PYTHONPATH="" alembic upgrade heads

# For quick local testing without Postgres (some migrations become no-ops):
export DATABASE_URL="sqlite:///test.db"
```

TimescaleDB and Bi-temporal notes:
- Migrations are resilient and idempotent for fundamentals.available_at, fundamentals.knowledge_date, indexes, and shares_outstanding. Safe to re-run.
- Optional TimescaleDB hypertable conversion for daily_bars is attempted on Postgres. Permissions may be required by your DB host.

Helpful CLI for infrastructure operations:
```bash
# TimescaleDB and validation utilities
python scripts/data_infra_cli.py --help
python scripts/data_infra_cli.py timescale --check
python scripts/data_infra_cli.py timescale --setup   # creates hypertables when possible
python scripts/data_infra_cli.py validate --full    # runs data validation pipeline
python scripts/data_infra_cli.py health-check       # institutional ingest health check
```

Known migration caveats:
- Multiple migration branches exist; always use "heads" not "head".
- On CI or containerized builds, SSL or permission issues can surface when running DDL (extensions). These are environment-specific and don’t indicate app bugs.


## Running the Application

### Streamlit Operator UI
```bash
# Requires DATABASE_URL
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Generic Data Dashboard
```bash
# Adapts to your schema; still requires DATABASE_URL to connect
streamlit run data_ingestion/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

### Task Queue (Celery + Redis)
Enable async pipelines and long-running jobs from the UI.

```bash
# Start Redis (choose one)
# macOS (Homebrew):
brew services start redis
# Docker:
docker run --rm -p 6379:6379 redis:7

# Start Celery worker from repo root
celery -A tasks worker --loglevel=info
# Windows or limited environments may prefer:
# celery -A tasks worker --pool=solo --loglevel=info
```

In the Streamlit sidebar, check "Use Task Queue (async)" to dispatch jobs (rebuild universe, ingest, features, train, backtest, generate trades, broker sync, full pipeline).

### Docker Deployment
```bash
# Build image (may fail in CI due to PyPI SSL issues unrelated to Dockerfile)
docker build -t smallcap-quant:latest .

# Web service (entrypoint supports SERVICE, APP_MODE, PORT)
# Default PORT in entrypoint is 10000
docker run --rm \
  -e DATABASE_URL="postgresql+psycopg://user:pass@host/db" \
  -e POLYGON_API_KEY=... -e APCA_API_KEY_ID=... -e APCA_API_SECRET_KEY=... \
  -e SERVICE=web -e APP_MODE=streamlit -e PORT=10000 \
  -p 10000:10000 smallcap-quant:latest

# Worker mode
docker run --rm -e SERVICE=worker -e DATABASE_URL="..." smallcap-quant:latest

# Cron/pipeline mode
docker run --rm -e SERVICE=cron   -e DATABASE_URL="..." smallcap-quant:latest
```

Docker build note: CI runners sometimes fail around ~40–50s with SSL certificate verification errors pulling wheels. This is environmental.


## Testing and Validation

### Unit/Smoke Tests
```bash
# Fast smoke tests that don’t require a running broker
python test_alpha_factory.py          # ~1.2s
python test_phase4_optimization.py    # ~1.1s

# Phase 2 infrastructure tests (many parts mock DB / set env)
python -m pytest test_phase2_infrastructure.py -v  # ~1.1s
python -m pytest test_integration_phase2.py -v     # imports + config checks; sets a mock DATABASE_URL
```

### Code Quality
```bash
# Lint key files
flake8 app.py config.py --count
# Full project lint (longer)
flake8 --max-line-length=160 .
```

Observed timings (reference): individual tests ~1.1–1.2s; pytest phase2 ~1.1s.


## Core Workflows

### End-of-Day Pipeline
```bash
# Full pipeline (requires API keys + DATABASE_URL)
python run_pipeline.py
```

### Individual Steps (from UI or CLI)
```bash
# Note: Some import functions may be missing from current implementation
# Universe + data (may have import issues)
# python -c "from data.universe import rebuild_universe; print(rebuild_universe()[:5])"
python -c "from data.ingest import ingest_bars_for_universe; ingest_bars_for_universe(730)"  # ~2y backfill
python -c "from data.fundamentals import fetch_fundamentals_for_universe; print(fetch_fundamentals_for_universe().shape)"

# Features / ML
python -m models.features
python -c "from models.ml import train_and_predict_all_models; outs=train_and_predict_all_models(); print(sum(len(v) for v in outs.values()) if outs else 0)"

# Trades / Broker
python -c "from trading.generate_trades import generate_today_trades; print(len(generate_today_trades()))"
python -c "from trading.broker import sync_trades_to_broker; print(sync_trades_to_broker([]))"  # pass real IDs
```


## Manual Validation Scenarios

ALWAYS run these after meaningful changes.

### Scenario 1: Basic App Startup
1. Set DATABASE_URL
2. Run Alembic migrations: `alembic upgrade heads` (or `PYTHONPATH="" alembic upgrade heads`)
3. Start Streamlit dashboard: `streamlit run data_ingestion/dashboard.py` (app.py has import issues)
4. Confirm UI loads at http://localhost:8501 and shows "Blank Capital Quant" dashboard

### Scenario 2: Async Task Queue
1. Start Redis and a Celery worker
2. In the Streamlit sidebar, enable "Use Task Queue (async)"
3. Dispatch Rebuild Universe → Ingest → Features → Train → Generate Trades
4. Monitor Task Monitoring panel for status and errors

### Scenario 3: Docker Validation
1. Build: `docker build -t test-quant .`
2. Run: `docker run --rm -e DATABASE_URL="sqlite:///test.db" -e PORT=10000 -p 10000:10000 test-quant`
3. Verify container starts; Alembic tries `upgrade heads` and continues even if migrations warn

### Scenario 4: Verification Scripts
1. `python scripts/verify_render_setup.py` → should print "All checks passed" items with ✅
2. `chmod +x scripts/entrypoint.sh && SERVICE=web APP_MODE=streamlit scripts/entrypoint.sh` → should start or error clearly on missing env


## Timing Expectations and Timeouts

NEVER CANCEL these operations:
- Dependency installation: 1–2 minutes. Timeout: 300+ seconds.
- Docker build: 5–10 minutes. Timeout: 1200+ seconds.
- Database migrations: 10–30 seconds (multi-head tolerant). Timeout: 120 seconds.
- Model training (with data): 5–30 minutes, dataset dependent. Timeout: 3600+ seconds.
- Full pipeline execution: 10–60 minutes, universe dependent. Timeout: 7200+ seconds.
- Individual tests: 1–2 seconds each. Timeout: 60 seconds.
- Verification scripts: < 1 second. Timeout: 30 seconds.


## Troubleshooting

Common issues and fixes:
1) "DATABASE_URL environment variable is required"
   - Set DATABASE_URL. For local smoke: `export DATABASE_URL="sqlite:///test.db"`

2) Alembic migration conflicts / multiple heads
   - Use `alembic upgrade heads` (not `head`). In CI, run with `PYTHONPATH=""` to avoid alembic package shadowing.

3) TimescaleDB extension or hypertable creation errors
   - Permission-related on managed Postgres. Use `scripts/data_infra_cli.py timescale --check` and proceed without Timescale if unavailable.

4) Docker build SSL errors in CI
   - Environment-specific PyPI SSL verification. Retry or pin mirrors; local builds typically succeed.

5) Import errors when DATABASE_URL not set
   - Many modules import db.py which requires DATABASE_URL. Set a local SQLite URL for imports/tests.

6) Celery/Redis connectivity
   - Ensure Redis is running and `REDIS_URL` matches. Start worker from repo root so `tasks` module resolves.

7) API quotas / rate limits
   - Reduce symbol universe or backfill window; monitor provider usage.

What works without a real Postgres DB:
- `test_alpha_factory.py`, `test_phase4_optimization.py`
- `test_phase2_infrastructure.py` and `test_integration_phase2.py` (they set/patch env/engine)
- Linting with flake8
- Docker build (CI SSL caveats)

Requires a database:
- Data ingestion and EOD pipeline
- Full backtests and trade generation with real data
- Streamlit features that query tables


## Key File Locations

- Configuration: `config.py` (env vars, toggles, logging)
- Main UI: `app.py` (Streamlit operator UI)
- Dashboard: `data_ingestion/dashboard.py`
- Pipeline: `run_pipeline.py` (checks, migrations, orchestration helpers)
- Database: `db.py` (SQLAlchemy models, engine, upserts)
- Models: `models/` (features, ML, regimes, transformers)
- Risk: `risk/` (covariance, sector neutralization, risk model)
- Portfolio: `portfolio/` (MVO optimizer and fallbacks)
- Trading: `trading/` (trade generation, broker sync)
- Data Infra: `data/validation.py`, `data/timescale.py`, `data/institutional_ingest.py`
- Tasks: `tasks/` (Celery tasks, dispatch utilities)
- Migrations: `alembic/` (use `heads`)
- Scripts: `scripts/entrypoint.sh`, `scripts/verify_render_setup.py`, `scripts/data_infra_cli.py`
- Tests: `test_*.py`


## Production Notes

- PostgreSQL + (optional) TimescaleDB extension for time-series
- Provider integrations: Polygon.io (PTI), Alpaca Markets, optional Tiingo
- Execution support: VWAP/TWAP and TCA hooks; portfolio construction with liquidity and beta constraints
- Walk-forward backtesting and validation tools
- Docker-first deployment; Render-friendly entrypoint with SERVICE/APP_MODE


## Validation Summary

- Last validated: September 2025
- Validation environment: CI/Ubuntu, Python 3.12

Working commands (verified):
- `pip install -r requirements.txt` (~1m 13s)
- `pip install -r requirements.extra.txt` (~1–2s)
- `python test_alpha_factory.py` (~1.2s)
- `python test_phase4_optimization.py` (~1.1s)
- `python -m pytest test_phase2_infrastructure.py -v` (~1.1s)
- `python -m pytest test_integration_phase2.py -v` (~1.1s, with DATABASE_URL)
- `streamlit run data_ingestion/dashboard.py` (starts successfully, connects when DATABASE_URL set)
- `uvicorn health_api:app --host 0.0.0.0 --port 8000` (API server starts successfully)
- `celery -A tasks worker --loglevel=info` (task queue ready with Redis)
- `python scripts/verify_render_setup.py` (~0.06s, passes checks)
- `alembic upgrade heads` (~2s, multi-head migration system)

Known issues to document:
- Multi-head Alembic history – use `alembic upgrade heads`
- CI Docker builds may fail early due to SSL certificate verification
- Some features require provider entitlements and may rate-limit in free tiers
- app.py has import issues: missing `rebuild_universe` function in data.universe module
- Some pipeline components have syntax errors (e.g., models/regime.py)
- Use data_ingestion/dashboard.py instead of app.py for testing UI functionality

⚠️ Trading is risky; paper trade first. Ensure API entitlements/quotas match your use case.