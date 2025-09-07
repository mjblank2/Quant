"""
Background Task Management Module

Provides robust background task management with timeout handling,
progress monitoring, and graceful cancellation for long-running requests.
"""
from __future__ import annotations
import logging
import time
import asyncio
import threading
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FutureTimeoutError

log = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class TaskInfo:
    """Information about a background task."""
    task_id: str
    name: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: Optional[str] = None
    timeout_seconds: int = 3600  # 1 hour default
    metadata: Dict[str, Any] = field(default_factory=dict)

class BackgroundTaskManager:
    """
    Manages background tasks with timeout handling and progress monitoring.
    """
    
    def __init__(self, max_workers: int = 4, default_timeout: int = 3600):
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, TaskInfo] = {}
        self.futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        self._shutdown = False
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def submit_task(self, task_id: str, name: str, func: Callable, 
                   timeout_seconds: Optional[int] = None, *args, **kwargs) -> TaskInfo:
        """
        Submit a task for background execution.
        
        Args:
            task_id: Unique identifier for the task
            name: Human-readable task name
            func: Function to execute
            timeout_seconds: Task timeout (uses default if None)
            *args, **kwargs: Arguments for the function
            
        Returns:
            TaskInfo object with task details
        """
        if timeout_seconds is None:
            timeout_seconds = self.default_timeout
        
        with self._lock:
            if task_id in self.tasks:
                existing_task = self.tasks[task_id]
                if existing_task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                    log.warning(f"Task {task_id} already running, returning existing task")
                    return existing_task
            
            # Create task info
            task_info = TaskInfo(
                task_id=task_id,
                name=name,
                status=TaskStatus.PENDING,
                created_at=datetime.now(),
                timeout_seconds=timeout_seconds
            )
            
            # Submit to executor
            future = self.executor.submit(self._execute_task, task_info, func, *args, **kwargs)
            
            self.tasks[task_id] = task_info
            self.futures[task_id] = future
            
            log.info(f"Submitted background task: {task_id} ({name}) with {timeout_seconds}s timeout")
            return task_info
    
    def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """Get current status of a task."""
        with self._lock:
            return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            task_id: Task to cancel
            
        Returns:
            True if cancellation was successful
        """
        with self._lock:
            if task_id not in self.tasks:
                return False
            
            task_info = self.tasks[task_id]
            future = self.futures.get(task_id)
            
            if task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False
            
            # Try to cancel the future
            cancelled = future.cancel() if future else False
            
            if cancelled or task_info.status == TaskStatus.PENDING:
                task_info.status = TaskStatus.CANCELLED
                task_info.completed_at = datetime.now()
                task_info.message = "Task cancelled by user"
                log.info(f"Cancelled task: {task_id}")
                return True
            
            return False
    
    def get_all_tasks(self) -> Dict[str, TaskInfo]:
        """Get all task information."""
        with self._lock:
            return dict(self.tasks)
    
    def cleanup_completed_tasks(self, older_than_hours: int = 24) -> int:
        """
        Clean up completed tasks older than specified hours.
        
        Args:
            older_than_hours: Remove tasks completed more than this many hours ago
            
        Returns:
            Number of tasks cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        cleaned_count = 0
        
        with self._lock:
            tasks_to_remove = []
            
            for task_id, task_info in self.tasks.items():
                if (task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                    task_info.completed_at and task_info.completed_at < cutoff_time):
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.tasks[task_id]
                if task_id in self.futures:
                    del self.futures[task_id]
                cleaned_count += 1
        
        if cleaned_count > 0:
            log.info(f"Cleaned up {cleaned_count} old completed tasks")
        
        return cleaned_count
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the task manager."""
        self._shutdown = True
        self.executor.shutdown(wait=wait)
    
    def _execute_task(self, task_info: TaskInfo, func: Callable, *args, **kwargs) -> Any:
        """Internal method to execute a task with timeout and error handling."""
        task_info.status = TaskStatus.RUNNING
        task_info.started_at = datetime.now()
        task_info.message = "Task started"
        
        try:
            log.info(f"Starting execution of task {task_info.task_id}")
            
            # Execute with timeout
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Update task completion
            task_info.status = TaskStatus.COMPLETED
            task_info.completed_at = datetime.now()
            task_info.progress = 100.0
            task_info.result = result
            task_info.message = f"Completed successfully in {execution_time:.2f}s"
            task_info.metadata['execution_time'] = execution_time
            
            log.info(f"Task {task_info.task_id} completed successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            task_info.status = TaskStatus.FAILED
            task_info.completed_at = datetime.now()
            task_info.error = str(e)
            task_info.message = f"Failed with error: {str(e)}"
            
            log.error(f"Task {task_info.task_id} failed: {e}")
            raise
    
    def _cleanup_worker(self) -> None:
        """Background worker to clean up old tasks and check timeouts."""
        while not self._shutdown:
            try:
                # Check for timed out tasks
                self._check_timeouts()
                
                # Clean up old completed tasks every hour
                self.cleanup_completed_tasks()
                
                # Sleep for 5 minutes before next check
                time.sleep(300)
                
            except Exception as e:
                log.error(f"Error in cleanup worker: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def _check_timeouts(self) -> None:
        """Check for tasks that have exceeded their timeout."""
        current_time = datetime.now()
        
        with self._lock:
            for task_id, task_info in self.tasks.items():
                if (task_info.status == TaskStatus.RUNNING and
                    task_info.started_at and
                    (current_time - task_info.started_at).total_seconds() > task_info.timeout_seconds):
                    
                    # Mark as timed out
                    task_info.status = TaskStatus.TIMEOUT
                    task_info.completed_at = current_time
                    task_info.error = f"Task exceeded timeout of {task_info.timeout_seconds} seconds"
                    task_info.message = "Task timed out"
                    
                    # Try to cancel the future
                    future = self.futures.get(task_id)
                    if future:
                        future.cancel()
                    
                    log.warning(f"Task {task_id} timed out after {task_info.timeout_seconds} seconds")

# Global task manager instance
_global_task_manager: Optional[BackgroundTaskManager] = None

def get_task_manager() -> BackgroundTaskManager:
    """Get the global task manager instance."""
    global _global_task_manager
    if _global_task_manager is None:
        _global_task_manager = BackgroundTaskManager()
    return _global_task_manager

def submit_background_task(task_id: str, name: str, func: Callable,
                          timeout_seconds: Optional[int] = None, 
                          *args, **kwargs) -> TaskInfo:
    """Submit a task to the global task manager."""
    return get_task_manager().submit_task(task_id, name, func, timeout_seconds, *args, **kwargs)

def get_task_status(task_id: str) -> Optional[TaskInfo]:
    """Get status of a task from the global task manager."""
    return get_task_manager().get_task_status(task_id)

def cancel_task(task_id: str) -> bool:
    """Cancel a task in the global task manager."""
    return get_task_manager().cancel_task(task_id)

def get_all_tasks() -> Dict[str, TaskInfo]:
    """Get all tasks from the global task manager."""
    return get_task_manager().get_all_tasks()

def cleanup_old_tasks(older_than_hours: int = 24) -> int:
    """Clean up old tasks from the global task manager."""
    return get_task_manager().cleanup_completed_tasks(older_than_hours)

# Specific helper functions for common pipeline tasks
def submit_pipeline_task(task_type: str, **kwargs) -> TaskInfo:
    """Submit a common pipeline task with appropriate timeout."""
    import uuid
    task_id = f"{task_type}_{uuid.uuid4().hex[:8]}"
    
    timeouts = {
        'universe_rebuild': 1800,    # 30 minutes
        'data_ingest': 3600,        # 1 hour
        'feature_build': 2700,      # 45 minutes
        'model_train': 7200,        # 2 hours
        'trade_generation': 600,    # 10 minutes
        'broker_sync': 300,         # 5 minutes
        'full_pipeline': 10800,     # 3 hours
    }
    
    timeout = timeouts.get(task_type, 3600)
    
    if task_type == 'universe_rebuild':
        from data.universe import rebuild_universe
        return submit_background_task(task_id, "Universe Rebuild", rebuild_universe, timeout)
    
    elif task_type == 'data_ingest':
        from data.ingest import ingest_bars_for_universe
        days = kwargs.get('days', 7)
        return submit_background_task(task_id, f"Data Ingestion ({days} days)", 
                                    ingest_bars_for_universe, timeout, days)
    
    elif task_type == 'feature_build':
        from models.features import build_features
        return submit_background_task(task_id, "Feature Building", build_features, timeout)
    
    elif task_type == 'model_train':
        from models.ml import train_and_predict_all_models
        return submit_background_task(task_id, "Model Training", train_and_predict_all_models, timeout)
    
    elif task_type == 'trade_generation':
        from enhanced_trade_generation import enhanced_generate_today_trades
        return submit_background_task(task_id, "Trade Generation", 
                                    enhanced_generate_today_trades, timeout)
    
    elif task_type == 'full_pipeline':
        from run_pipeline import main
        sync_broker = kwargs.get('sync_broker', False)
        return submit_background_task(task_id, f"Full Pipeline (sync={sync_broker})", 
                                    main, timeout, sync_broker)
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")