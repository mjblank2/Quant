# tasks/celery_app.py
from __future__ import annotations
import os
from celery import Celery
from config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

# Create Celery instance
celery_app = Celery('quant_system')

# Configure Celery
celery_app.conf.update(
    broker_url=CELERY_BROKER_URL,
    result_backend=CELERY_RESULT_BACKEND,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    result_expires=3600,  # Results expire after 1 hour
    task_track_started=True,
    task_annotations={
        '*': {'rate_limit': '10/s'}
    },
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=50,
)

# Auto-discover tasks in this module
celery_app.autodiscover_tasks(['tasks'])

if __name__ == '__main__':
    celery_app.start()