#!/usr/bin/env python3
"""
Simple script to start a Celery worker for the Quant system
Usage: python start_celery_worker.py
"""
from __future__ import annotations
import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - celery - %(levelname)s - %(message)s"
)

def main():
    try:
        from tasks.celery_app import celery_app
        
        # Configure worker
        worker_args = [
            'worker',
            '--loglevel=info',
            '--concurrency=2',
            '--task-events',
            '--without-heartbeat',
            '--without-gossip',
            '--without-mingle',
        ]
        
        print("Starting Celery worker...")
        print("Make sure Redis is running on the configured URL")
        print("Press Ctrl+C to stop")
        
        celery_app.start(worker_args)
        
    except ImportError as e:
        print(f"Failed to import Celery tasks: {e}")
        print("Make sure you have installed celery[redis] and Redis is configured")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopping Celery worker...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting Celery worker: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()