# Task Queue Implementation

This document describes the newly implemented task queue infrastructure that decouples Streamlit from long-running operations.

## Cron vs. Task Queue

- Cron jobs (scheduled operations) should run the Python modules directly inside the container. This is the simplest and most reliable approach and does not require Celery or Redis.
- The task queue (Celery + Redis) is intended for asynchronous operations triggered from the Streamlit UI or the FastAPI service.

Recommended cron commands (also wired in render.yaml):
- Refresh universe: bash -lc scripts/cron_universe.sh
- Ingest bars: bash -lc scripts/cron_ingest.sh (honors DAYS env; default 7)
- EOD pipeline: bash -lc scripts/cron_eod_pipeline.sh

## Architecture Overview

The system now supports two modes:

### 1. Synchronous Mode (Original)
- Direct execution of tasks in the Streamlit UI
- Blocks the UI during operation
- Simple but not suitable for production

### 2. Asynchronous Mode (New)
- Tasks dispatched to a Redis-backed Celery queue
- UI remains responsive
- Background workers process tasks
- Real-time status monitoring

## Components

### Task Queue Infrastructure
- **Celery**: Distributed task queue system
- **Redis**: Message broker and result backend
- **TaskStatus Model**: Database tracking of task states

### Core Tasks
All major operations have been converted to Celery tasks:
- `rebuild_universe_task`: Universe rebuilding
- `ingest_market_data_task`: Market data ingestion  
- `ingest_fundamentals_task`: Fundamentals data ingestion
- `build_features_task`: Feature engineering
- `train_and_predict_task`: Model training and prediction
- `run_backtest_task`: Walk-forward backtesting
- `generate_trades_task`: Trade generation
- `sync_broker_task`: Broker synchronization
- `run_full_pipeline_task`: Complete pipeline execution

### UI Enhancements
- Toggle between sync/async modes
- Real-time task monitoring
- Progress tracking
- Error reporting

## Setup Instructions

### 1. Install Dependencies
The required dependencies are already in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Start Redis
Install and start Redis server:
```bash
# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis-server

# macOS
brew install redis
brew services start redis

# Docker
docker run -d -p 6379:6379 redis:alpine
```

### 3. Configure Environment
Set Redis URL (optional, defaults to localhost):
```bash
export REDIS_URL="redis://localhost:6379/0"
export CELERY_BROKER_URL="redis://localhost:6379/0"
export CELERY_RESULT_BACKEND="redis://localhost:6379/0"
```

### 4. Start Celery Worker
Start a Celery worker to process tasks:
```bash
# Option 1: Using the helper script
python start_celery_worker.py

# Option 2: Using the worker module
python -m jobs.worker celery

# Option 3: Direct Celery command
celery -A tasks.celery_app worker --loglevel=info --concurrency=2
```

### 5. Run Streamlit
Start the Streamlit application:
```bash
streamlit run app.py
```

## Usage

### In the Streamlit UI
1. Check "Use Task Queue (async)" to enable async mode
2. Click any operation button (e.g., "Backfill Market Data")
3. Task is dispatched immediately - UI remains responsive
4. Monitor progress in the "Task Monitoring" section
5. View real-time status updates

### Task Status States
- **PENDING**: Task queued, waiting for worker
- **STARTED**: Task picked up by worker, in progress
- **SUCCESS**: Task completed successfully
- **FAILURE**: Task failed with error
- **RETRY**: Task failed but will be retried

## Production Deployment

### Docker Deployment
The system supports containerized deployment:

1. **Web Service**: Streamlit UI
   ```bash
   SERVICE=web APP_MODE=operator
   ```

2. **Worker Service**: Celery workers
   ```bash
   SERVICE=worker WORKER_TASK=celery
   ```

3. **Redis Service**: Message broker
   ```bash
   # Use managed Redis (e.g., Redis Cloud, AWS ElastiCache)
   # Or deploy Redis container
   ```

### Scaling
- **Horizontal**: Deploy multiple worker containers
- **Vertical**: Increase worker concurrency
- **Redis**: Use Redis cluster for high availability

## Monitoring & Observability

### Built-in Monitoring
- Task status tracking in database
- Real-time UI updates
- Error logging and reporting

### External Tools
- **Flower**: Celery monitoring dashboard
  ```bash
  pip install flower
  celery -A tasks.celery_app flower
  ```
- **Redis monitoring**: RedisInsight, redis-cli
- **Application logs**: Structured logging throughout

## Backward Compatibility

The implementation maintains full backward compatibility:
- Original synchronous mode still available
- Existing scripts (`run_pipeline.py`) unchanged
- No breaking changes to core business logic

## Benefits Achieved

✅ **UI Responsiveness**: No more blocking operations  
✅ **Scalability**: Multiple workers can process tasks  
✅ **Reliability**: Automatic retries and error handling  
✅ **Monitoring**: Real-time status and progress tracking  
✅ **Production Ready**: Proper task queue infrastructure  

This implementation addresses the key limitation identified in the strategic roadmap: "using Streamlit as the orchestrator is fragile for production".