# ---- base image ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHOPTS="-u"

# System deps (Postgres driver, build tools for scientific stack)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- app files ----
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install Python deps (includes alembic, streamlit, xgboost, cvxpy, etc.)
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# Copy the rest of the repo
COPY . /app

# Add entrypoint
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Render sets $PORT; Streamlit will bind to it from the entrypoint
CMD ["/entrypoint.sh"]
