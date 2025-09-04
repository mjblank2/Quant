
try:
    from celery import Celery  # type: ignore
except Exception:
    Celery = None

if Celery is not None:
    celery_app = Celery('blank_capital', broker='redis://localhost:6379/0')
else:
    celery_app = None
