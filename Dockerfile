# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (pg headers for psycopg, build tools for wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \    build-essential \    libpq-dev \    curl ca-certificates \ && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements (main + extra)
COPY requirements.txt /app/requirements.txt
# HOTFIX: malformed Alembic pin in your file
RUN sed -i -E 's/(^|[[:space:]])alembic>=([^,[:space:]]+)</\1alembic>=\2,</' /app/requirements.txt || true

# Install main requirements
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Add extra requirements that ensure DB driver
COPY requirements.extra.txt /app/requirements.extra.txt
RUN pip install -r /app/requirements.extra.txt

# Copy the rest of the app
COPY . /app

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Make sure entrypoint is executable if present
RUN test -f /app/scripts/entrypoint.sh && chmod +x /app/scripts/entrypoint.sh || true

EXPOSE 8000

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
