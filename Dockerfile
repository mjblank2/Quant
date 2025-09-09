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

# Copy the rest of the app
COPY . /app

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
