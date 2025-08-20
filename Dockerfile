# syntax=docker/dockerfile:1

### --- Builder Stage --- ###
FROM python:3.11.9-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ONLY_BINARY=:all:         # <- never compile from source

WORKDIR /app

# Minimal toolchain (kept in builder only)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# Bring in the app
COPY . .

### --- Final Runtime Stage --- ###
FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # avoid CPU oversubscription when also using Gunicorn threads
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1

# Small runtime libs commonly needed by wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libstdc++6 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed site-packages and the app
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Flask/WSGI -> use a standard Gunicorn worker (gthread)
CMD gunicorn \
  -k gthread \
  -w ${WEB_CONCURRENCY:-2} \
  --threads ${THREADS:-4} \
  --timeout ${TIMEOUT:-120} \
  --keep-alive ${KEEP_ALIVE:-5} \
  --max-requests ${MAX_REQUESTS:-1000} \
  --max-requests-jitter ${MAX_REQUESTS_JITTER:-100} \
  --access-logfile - \
  --error-logfile - \
  -b 0.0.0.0:${PORT:-8080} \
  data_ingestion.main:app

