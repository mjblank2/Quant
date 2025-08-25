#!/usr/bin/env bash
set -euo pipefail

# Defaults
: "${PORT:=8000}"
: "${SERVICE:=web}"              # web | worker | cron
: "${APP_MODE:=streamlit}"       # streamlit | api
: "${RUN_MIGRATIONS:=1}"         # 1 to run alembic upgrade head on start

echo "[entrypoint] SERVICE=${SERVICE} APP_MODE=${APP_MODE} PORT=${PORT}"

# Best-effort migrations if configured
if [[ "${RUN_MIGRATIONS}" = "1" ]]; then
  if [[ -n "${DATABASE_URL:-}" ]]; then
    echo "[entrypoint] Running Alembic migrations..."
    # Render often uses postgres://; SQLAlchemy 2 wants postgresql+psycopg://
    if [[ "${DATABASE_URL}" == postgres://* ]]; then
      export DATABASE_URL="${DATABASE_URL/postgres:\/\//postgresql+psycopg://}"
    fi
    if command -v alembic >/dev/null 2>&1; then
      alembic upgrade head || { echo "[entrypoint] WARNING: alembic upgrade failed, continuing"; }
    else
      echo "[entrypoint] WARNING: alembic not found; skipping migrations"
    fi
  else
    echo "[entrypoint] DATABASE_URL not set; skipping migrations"
  fi
fi

case "${SERVICE}" in
  web)
    case "${APP_MODE}" in
      streamlit)
        echo "[entrypoint] Starting Streamlit dashboard on port ${PORT}"
        exec streamlit run data_ingestion/dashboard.py --server.port "${PORT}" --server.address 0.0.0.0
        ;;
      api)
        # Change 'app.main:app' to your FastAPI app import path if different
        echo "[entrypoint] Starting API (uvicorn) on port ${PORT}"
        exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}"
        ;;
      *)
        echo "[entrypoint] Unknown APP_MODE='${APP_MODE}'"; exit 1;;
    esac
    ;;
  worker)
    echo "[entrypoint] Starting worker..."
    # Replace with your worker command
    exec python -m jobs.worker
    ;;
  cron)
    echo "[entrypoint] Running scheduled task..."
    # Replace with your cron task or dispatcher
    exec python -m jobs.daily
    ;;
  *)
    echo "[entrypoint] Unknown SERVICE='${SERVICE}'"; exit 1;;
esac
