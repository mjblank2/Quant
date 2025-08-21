# syntax=docker/dockerfile:1

### --- Builder Stage --- ###
FROM python:3.11.9-bookworm AS builder

# Set environment variables for a clean and efficient build process.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build-essential (needed if any packages must compile C extensions).
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
COPY requirements.txt .
# Optimized: Prefer pre-built binaries (wheels); fallback to source build if necessary.
RUN pip install --upgrade pip setuptools wheel \
 && (pip install --only-binary=:all: -r requirements.txt || pip install -r requirements.txt)

# Copy the application code into the builder stage.
COPY . .


### --- Final Runtime Stage --- ###
FROM python:3.11.9-slim-bookworm

# Set environment variables for runtime.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install only necessary runtime libraries (libgomp1 is often needed by numpy/pandas).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages and the application code from the builder stage.
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

# Create a non-root user for security and set appropriate permissions.
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port the app will run on.
EXPOSE 8080

# Run the application using gunicorn.
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
