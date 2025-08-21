# syntax=docker/dockerfile:1

### --- Builder Stage --- ###
FROM python:3.11.9-bookworm AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PIP_ONLY_BINARY=:all:
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
COPY . .

### --- Final Runtime Stage --- ###
FROM python:3.11.9-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Streamlit defaults to 8501; Render sets $PORT dynamically.
EXPOSE 8501
CMD streamlit run data_ingestion/dashboard.py \
  --server.headless=true \
  --server.address=0.0.0.0 \
  --server.port=${PORT:-8501} \
  --server.enableCORS=false \
  --browser.gatherUsageStats=false

