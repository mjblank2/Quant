# tasks/core_tasks.py
from __future__ import annotations
import logging
from datetime import datetime
from celery import current_task
from sqlalchemy import text
from db import engine, SessionLocal, TaskStatus, upsert_dataframe
from .celery_app import celery_app

log = logging.getLogger(__name__)

def update_task_status(task_id: str, status: str, progress: int = 0, result: dict = None, error_message: str = None):
    """Update task status in database"""
    try:
        with SessionLocal() as session:
            task_status = session.get(TaskStatus, task_id)
            if not task_status:
                task_status = TaskStatus(
                    task_id=task_id,
                    task_name=current_task.name,
                    status=status,
                    progress=progress
                )
                session.add(task_status)
            else:
                task_status.status = status
                task_status.progress = progress
                if status == 'STARTED' and not task_status.started_at:
                    task_status.started_at = datetime.utcnow()
                elif status in ('SUCCESS', 'FAILURE'):
                    task_status.completed_at = datetime.utcnow()
            
            if result:
                task_status.result = result
            if error_message:
                task_status.error_message = error_message
                
            session.commit()
    except Exception as e:
        log.error(f"Failed to update task status: {e}")

@celery_app.task(bind=True)
def rebuild_universe_task(self):
    """Rebuild the universe of stocks"""
    task_id = self.request.id
    update_task_status(task_id, 'STARTED', 0)
    
    try:
        from data.universe import rebuild_universe
        update_task_status(task_id, 'STARTED', 25)
        
        universe = rebuild_universe()
        universe_size = len(universe) if universe is not None else 0
        
        update_task_status(task_id, 'SUCCESS', 100, {'universe_size': universe_size})
        return {'universe_size': universe_size}
        
    except Exception as e:
        error_msg = f"Universe rebuild failed: {str(e)}"
        update_task_status(task_id, 'FAILURE', 0, error_message=error_msg)
        raise

@celery_app.task(bind=True)
def ingest_market_data_task(self, days: int = 730):
    """Ingest market data for the universe"""
    task_id = self.request.id
    update_task_status(task_id, 'STARTED', 0)
    
    try:
        from data.ingest import ingest_bars_for_universe
        update_task_status(task_id, 'STARTED', 25)
        
        ingest_bars_for_universe(days)
        
        # Get count of ingested data
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM daily_bars")).scalar()
        
        update_task_status(task_id, 'SUCCESS', 100, {'days': days, 'total_bars': count})
        return {'days': days, 'total_bars': count}
        
    except Exception as e:
        error_msg = f"Market data ingestion failed: {str(e)}"
        update_task_status(task_id, 'FAILURE', 0, error_message=error_msg)
        raise

@celery_app.task(bind=True)
def ingest_fundamentals_task(self):
    """Ingest fundamentals data"""
    task_id = self.request.id
    update_task_status(task_id, 'STARTED', 0)
    
    try:
        from data.fundamentals import fetch_fundamentals_for_universe
        update_task_status(task_id, 'STARTED', 25)
        
        df = fetch_fundamentals_for_universe()
        rows_count = len(df) if df is not None else 0
        
        update_task_status(task_id, 'SUCCESS', 100, {'fundamentals_rows': rows_count})
        return {'fundamentals_rows': rows_count}
        
    except Exception as e:
        error_msg = f"Fundamentals ingestion failed: {str(e)}"
        update_task_status(task_id, 'FAILURE', 0, error_message=error_msg)
        raise

@celery_app.task(bind=True)
def build_features_task(self):
    """Build features incrementally"""
    task_id = self.request.id
    update_task_status(task_id, 'STARTED', 0)
    
    try:
        from models.features import build_features
        update_task_status(task_id, 'STARTED', 25)
        
        features = build_features()
        feature_count = len(features) if features is not None else 0
        
        update_task_status(task_id, 'SUCCESS', 100, {'new_feature_rows': feature_count})
        return {'new_feature_rows': feature_count}
        
    except Exception as e:
        error_msg = f"Feature building failed: {str(e)}"
        update_task_status(task_id, 'FAILURE', 0, error_message=error_msg)
        raise

@celery_app.task(bind=True)
def train_and_predict_task(self):
    """Train models and generate predictions"""
    task_id = self.request.id
    update_task_status(task_id, 'STARTED', 0)
    
    try:
        from models.ml import train_and_predict_all_models
        update_task_status(task_id, 'STARTED', 25)
        
        outs = train_and_predict_all_models()
        total_predictions = sum(len(v) for v in outs.values()) if outs else 0
        model_count = len(outs) if outs else 0
        
        update_task_status(task_id, 'SUCCESS', 100, {
            'models_trained': model_count,
            'total_predictions': total_predictions
        })
        return {'models_trained': model_count, 'total_predictions': total_predictions}
        
    except Exception as e:
        error_msg = f"Training/prediction failed: {str(e)}"
        update_task_status(task_id, 'FAILURE', 0, error_message=error_msg)
        raise

@celery_app.task(bind=True)
def run_backtest_task(self):
    """Run walk-forward backtest"""
    task_id = self.request.id
    update_task_status(task_id, 'STARTED', 0)
    
    try:
        from models.ml import run_walkforward_backtest
        update_task_status(task_id, 'STARTED', 25)
        
        bt = run_walkforward_backtest()
        backtest_rows = len(bt) if bt is not None else 0
        
        update_task_status(task_id, 'SUCCESS', 100, {'backtest_rows': backtest_rows})
        return {'backtest_rows': backtest_rows}
        
    except Exception as e:
        error_msg = f"Backtest failed: {str(e)}"
        update_task_status(task_id, 'FAILURE', 0, error_message=error_msg)
        raise

@celery_app.task(bind=True)
def generate_trades_task(self):
    """Generate today's trades"""
    task_id = self.request.id
    update_task_status(task_id, 'STARTED', 0)
    
    try:
        from trading.generate_trades import generate_today_trades
        update_task_status(task_id, 'STARTED', 25)
        
        trades = generate_today_trades()
        trade_count = len(trades) if trades is not None else 0
        
        update_task_status(task_id, 'SUCCESS', 100, {'trades_generated': trade_count})
        return {'trades_generated': trade_count}
        
    except Exception as e:
        error_msg = f"Trade generation failed: {str(e)}"
        update_task_status(task_id, 'FAILURE', 0, error_message=error_msg)
        raise

@celery_app.task(bind=True)
def sync_broker_task(self, trade_ids: list[int] = None):
    """Sync trades with broker"""
    task_id = self.request.id
    update_task_status(task_id, 'STARTED', 0)
    
    try:
        from trading.broker import sync_trades_to_broker
        update_task_status(task_id, 'STARTED', 25)
        
        if not trade_ids:
            # Get recent generated trades
            with engine.connect() as conn:
                recent = conn.execute(text(
                    "SELECT id FROM trades WHERE status='generated' ORDER BY id DESC LIMIT 2000"
                )).fetchall()
                trade_ids = [row[0] for row in recent]
        
        if not trade_ids:
            update_task_status(task_id, 'SUCCESS', 100, {'message': 'No trades to sync'})
            return {'message': 'No trades to sync'}
        
        result = sync_trades_to_broker(trade_ids)
        synced_count = len(result) if result else 0
        
        update_task_status(task_id, 'SUCCESS', 100, {'trades_synced': synced_count})
        return {'trades_synced': synced_count}
        
    except Exception as e:
        error_msg = f"Broker sync failed: {str(e)}"
        update_task_status(task_id, 'FAILURE', 0, error_message=error_msg)
        raise

@celery_app.task(bind=True)
def run_full_pipeline_task(self, sync_broker: bool = False):
    """Run the full end-of-day pipeline"""
    task_id = self.request.id
    update_task_status(task_id, 'STARTED', 0)
    
    try:
        from run_pipeline import main
        update_task_status(task_id, 'STARTED', 25)
        
        result = main(sync_broker=sync_broker)
        
        update_task_status(task_id, 'SUCCESS', 100, {
            'pipeline_success': result,
            'sync_broker': sync_broker
        })
        return {'pipeline_success': result, 'sync_broker': sync_broker}
        
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        update_task_status(task_id, 'FAILURE', 0, error_message=error_msg)
        raise