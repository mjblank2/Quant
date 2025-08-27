#!/usr/bin/env bash
set -Eeuo pipefail

# Defaults
SERVICE="${SERVICE:-web}"          # web | worker | cron
APP_MODE="${APP_MODE:-streamlit}"  # streamlit | operator | api
PORT="${PORT:-10000}"

echo "[entrypoint] SERVICE=${SERVICE} APP_MODE=${APP_MODE} PORT=${PORT}"

# Always run Alembic migrations first (if present)
if [[ -f "alembic.ini" && -d "alembic" ]]; then
  echo "[entrypoint] Running Alembic upgrade..."
  if alembic upgrade head; then
    echo "[entrypoint] Alembic upgrade succeeded"
  else
    echo "[entrypoint] Alembic upgrade failed (continuing)"
  fi
fi

# If a command was provided by the runtime (e.g., Render cron dockerCommand),
# run it instead of our built-in modes.
if [[ "$#" -gt 0 ]]; then
  echo "[entrypoint] Command override detected: $*"
  exec "$@"
fi

case "${SERVICE}" in
  web)
    case "${APP_MODE}" in
      streamlit)
        echo "[entrypoint] Starting Streamlit dashboard on port ${PORT}"
        exec streamlit run data_ingestion/dashboard.py \
          --server.port "${PORT}" --server.address 0.0.0.0
        ;;
      operator)
        echo "[entrypoint] Starting Streamlit operator app on port ${PORT}"
        exec streamlit run app.py \
          --server.port "${PORT}" --server.address 0.0.0.0
        ;;
      api)
        echo "[entrypoint] Starting FastAPI on port ${PORT}"
        exec uvicorn health_api:app --host 0.0.0.0 --port "${PORT}"
        ;;
      *)
        echo "[entrypoint] Unknown APP_MODE='${APP_MODE}'\n Available modes: streamlit, operator, api"; exit 1;;
    esac
    ;;
  worker)
    echo "[entrypoint] Starting worker..."
    if [[ "${WORKER_TASK:-idle}" == "celery" ]]; then
      echo "[entrypoint] Starting Celery worker..."
      exec python -m jobs.worker celery
    else
      exec python -m jobs.worker
    fi
    ;;
  cron)
    # Default cron behavior (when no dockerCommand override)
    echo "[entrypoint] Running EOD pipeline..."
    exec python run_pipeline.py
    ;;
  *)
    echo "[entrypoint] Unknown SERVICE='${SERVICE}'"; exit 1;;
esac
