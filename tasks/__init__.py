# tasks/__init__.py
from .celery_app import celery_app
from .core_tasks import (
    rebuild_universe_task,
    ingest_market_data_task,
    ingest_fundamentals_task,
    build_features_task,
    train_and_predict_task,
    run_backtest_task,
    generate_trades_task,
    sync_broker_task,
    run_full_pipeline_task
)

__all__ = [
    'celery_app',
    'rebuild_universe_task',
    'ingest_market_data_task',
    'ingest_fundamentals_task',
    'build_features_task',
    'train_and_predict_task',
    'run_backtest_task',
    'generate_trades_task',
    'sync_broker_task',
    'run_full_pipeline_task'
]