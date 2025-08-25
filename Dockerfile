# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (pg headers for psycopg, build tools for wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy the rest of the app
COPY . /app

# Non-root user for safety
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Ensure entrypoint exists and is executable if present
RUN test -f /app/scripts/entrypoint.sh && chmod +x /app/scripts/entrypoint.sh || true

EXPOSE 8000

# Start through our entrypoint (lives inside repo under scripts/)
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
