# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (single line to avoid line-continuation issues)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and fix requirements (PEP 440 comma hotfix for Alembic)
COPY requirements.txt /app/requirements.txt
RUN sed -i -E 's/(^|[[:space:]])alembic>=([^,[:space:]]+)</\1alembic>=\2,</' /app/requirements.txt || true

# Install main requirements
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Ensure DB driver
COPY requirements.extra.txt /app/requirements.extra.txt
RUN pip install -r /app/requirements.extra.txt

# Copy the application source selectively.  Avoid copying the entire repository
# (including test data, caches, and other artifacts) into the image.  This
# reduces the build context size and prevents cache extraction errors.
COPY data /app/data
COPY data_ingestion /app/data_ingestion
COPY features /app/features
COPY models /app/models
COPY ml /app/ml
COPY portfolio /app/portfolio
COPY tasks /app/tasks
COPY scripts /app/scripts

# Include missing directories and files
COPY jobs /app/jobs
COPY utils /app/utils
COPY app.py /app/app.py
COPY health_api.py /app/health_api.py
COPY market_calendar.py /app/market_calendar.py
# Individual top‑level modules and configuration
COPY worker.py /app/worker.py
COPY run_pipeline.py /app/run_pipeline.py
COPY config.py /app/config.py
COPY db.py /app/db.py
COPY alembic /app/alembic
COPY alembic.ini /app/alembic.ini

# Include requirements files again for clarity (they are already copied above butC
# re-copying here is harmless and ensures they exist in the final image).
COPY utils_http.py /app/utils_http.py
COPY utils_http_async.py /app/utils_http_async.py
COPY utils_logging.py /app/utils_logging.py
COPY requirements.txt /app/requirements.txt
COPY requirements.extra.txt /app/requirements.extra.txt

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Make scripts executable if present
RUN test -f /app/scripts/entrypoint.sh && chmod +x /app/scripts/entrypoint.sh || true
RUN test -f /app/scripts/cron_universe.sh && chmod +x /app/scripts/cron_universe.sh || true
RUN test -f /app/scripts/cron_ingest.sh && chmod +x /app/scripts/cron_ingest.sh || true
RUN test -f /app/scripts/cron_eod_pipeline.sh && chmod +x /app/scripts/cron_eod_pipeline.sh || true

# Align exposed port with Streamlit default in entrypoint (PORT=10000)
EXPOSE 10000

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
