# tasks/task_utils.py
from __future__ import annotations
import uuid
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy import text
from db import engine, SessionLocal, TaskStatus

def dispatch_task(task_func, *args, **kwargs) -> str:
    """Dispatch a task and return task ID"""
    task_result = task_func.delay(*args, **kwargs)
    return task_result.id

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status from database"""
    try:
        with SessionLocal() as session:
            task_status = session.get(TaskStatus, task_id)
            if not task_status:
                return None
            
            return {
                'task_id': task_status.task_id,
                'task_name': task_status.task_name,
                'status': task_status.status,
                'progress': task_status.progress,
                'created_at': task_status.created_at,
                'started_at': task_status.started_at,
                'completed_at': task_status.completed_at,
                'result': task_status.result,
                'error_message': task_status.error_message,
            }
    except Exception:
        return None

def get_recent_tasks(limit: int = 20) -> list[Dict[str, Any]]:
    """Get recent tasks from database"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT task_id, task_name, status, progress, created_at, started_at, completed_at, error_message
                FROM task_status 
                ORDER BY created_at DESC 
                LIMIT :limit
            """), {"limit": limit})
            
            tasks = []
            for row in result:
                tasks.append({
                    'task_id': row[0],
                    'task_name': row[1],
                    'status': row[2],
                    'progress': row[3],
                    'created_at': row[4],
                    'started_at': row[5],
                    'completed_at': row[6],
                    'error_message': row[7],
                })
            return tasks
    except Exception:
        return []

def cleanup_old_tasks(days_old: int = 7):
    """Clean up old task records"""
    try:
        cutoff = datetime.utcnow() - timedelta(days=days_old)
        with engine.connect() as conn:
            conn.execute(text("""
                DELETE FROM task_status 
                WHERE created_at < :cutoff AND status IN ('SUCCESS', 'FAILURE')
            """), {"cutoff": cutoff})
            conn.commit()
    except Exception:
        pass