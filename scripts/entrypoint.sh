#!/usr/bin/env bash
set -Eeuo pipefail

# Defaults
SERVICE="${SERVICE:-web}"          # web | worker | cron
APP_MODE="${APP_MODE:-streamlit}"  # streamlit | operator | api
PORT="${PORT:-10000}"

echo "[entrypoint] SERVICE=${SERVICE} APP_MODE=${APP_MODE} PORT=${PORT}"
echo "[entrypoint] Environment check: DATABASE_URL=${DATABASE_URL:+SET} PYTHONPATH=${PYTHONPATH:-UNSET}"

# Function to check environment variables for different services
check_environment() {
    case "${SERVICE}" in
        web|worker|cron)
            if [[ -z "${DATABASE_URL:-}" ]]; then
                echo "[entrypoint] ❌ ERROR: DATABASE_URL environment variable is required for ${SERVICE} service"
                exit 1
            fi
            echo "[entrypoint] ✅ DATABASE_URL is configured"
            ;;
    esac
}

# Always run Alembic migrations first (if present)
if [[ -f "alembic.ini" && -d "alembic" ]]; then
  echo "[entrypoint] Running Alembic upgrade..."
  if alembic upgrade head; then
    echo "[entrypoint] ✅ Alembic upgrade succeeded"
  else
    echo "[entrypoint] ⚠️ Alembic upgrade failed (continuing anyway)"
  fi
fi

# If a command was provided by the runtime (e.g., Render cron dockerCommand),
# run it instead of our built-in modes.
if [[ "$#" -gt 0 ]]; then
  echo "[entrypoint] 🚀 Command override detected: $*"
  echo "[entrypoint] Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  
  # Check environment before running command
  check_environment
  
  # Run the command and capture exit status
  set +e
  "$@"
  exit_code=$?
  set -e
  
  echo "[entrypoint] Command completed with exit code: ${exit_code}"
  if [[ ${exit_code} -eq 0 ]]; then
    echo "[entrypoint] ✅ Command execution successful"
  else
    echo "[entrypoint] ❌ Command execution failed with exit code ${exit_code}"
  fi
  
  exit ${exit_code}
fi

case "${SERVICE}" in
  web)
    check_environment
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
        echo "[entrypoint] ❌ Unknown APP_MODE='${APP_MODE}'\n Available modes: streamlit, operator, api"; exit 1;;
    esac
    ;;
  worker)
    check_environment
    echo "[entrypoint] Starting worker..."
    if [[ "${WORKER_TASK:-idle}" == "celery" ]]; then
      echo "[entrypoint] Starting Celery worker..."
      exec python -m jobs.worker celery
    else
      exec python -m jobs.worker
    fi
    ;;
  cron)
    check_environment
    # Default cron behavior (when no dockerCommand override)
    echo "[entrypoint] 🕰️ Running EOD pipeline..."
    echo "[entrypoint] Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    
    # Run the pipeline and capture exit status
    set +e
    python run_pipeline.py
    exit_code=$?
    set -e
    
    echo "[entrypoint] Pipeline completed with exit code: ${exit_code}"
    if [[ ${exit_code} -eq 0 ]]; then
      echo "[entrypoint] ✅ Pipeline execution successful"
    else
      echo "[entrypoint] ❌ Pipeline execution failed with exit code ${exit_code}"
    fi
    
    exit ${exit_code}
    ;;
  *)
    echo "[entrypoint] ❌ Unknown SERVICE='${SERVICE}'"; exit 1;;
esac
