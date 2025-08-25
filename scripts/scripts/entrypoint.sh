#!/usr/bin/env bash
set -euo pipefail

# Defaults
: "${PORT:=8000}"
: "${SERVICE:=web}"              # web | worker | cron
: "${APP_MODE:=streamlit}"       # streamlit | api
: "${RUN_MIGRATIONS:=1}"         # 1 to run alembic upgrade head on start

echo "[entrypoint] SERVICE=${SERVICE} APP_MODE=${APP_MODE} PORT=${PORT}"

run_migrations() {
  if [[ "${RUN_MIGRATIONS}" != "1" ]]; then
    echo "[entrypoint] RUN_MIGRATIONS=0; skipping migrations"
    return 0
  fi

  if [[ -z "${DATABASE_URL:-}" ]]; then
    echo "[entrypoint] DATABASE_URL not set; skipping migrations"
    return 0
  fi

  # Normalize DSN for SQLAlchemy 2.x
  if [[ "${DATABASE_URL}" == postgres://* ]]; then
    export DATABASE_URL="${DATABASE_URL/postgres:\/\//postgresql+psycopg://}"
  fi

  if ! command -v alembic >/dev/null 2>&1; then
    echo "[entrypoint] WARNING: alembic not found; skipping migrations"
    return 0
  fi

  set +e
  echo "[entrypoint] Running Alembic upgrade..."
  UPGRADE_LOG="$(alembic upgrade head 2>&1)"
  STATUS=$?
  set -e

  if [[ $STATUS -eq 0 ]]; then
    echo "[entrypoint] Alembic upgrade succeeded"
    return 0
  fi

  echo "[entrypoint] Alembic upgrade failed; inspecting error..."
  echo "$UPGRADE_LOG"

  # If it's a DuplicateTable error (schema already exists), baseline then retry once
  if echo "$UPGRADE_LOG" | grep -qiE 'DuplicateTable|already exists'; then
    echo "[entrypoint] Detected pre-existing schema. Stamping DB to current head, then retrying upgrade once."
    alembic stamp head || { echo "[entrypoint] WARNING: stamp head failed"; return 0; }
    alembic upgrade head || { echo "[entrypoint] WARNING: second upgrade attempt failed; continuing anyway"; return 0; }
    echo "[entrypoint] Alembic upgrade succeeded after stamp."
    return 0
  fi

  echo "[entrypoint] WARNING: alembic upgrade failed with non-duplicate error; continuing"
  return 0
}

run_service() {
  case "${SERVICE}" in
    web)
      case "${APP_MODE}" in
        streamlit)
          echo "[entrypoint] Starting Streamlit dashboard on port ${PORT}"
          exec streamlit run data_ingestion/dashboard.py --server.port "${PORT}" --server.address 0.0.0.0
          ;;
        api)
          echo "[entrypoint] Starting API (uvicorn) on port ${PORT}"
          exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}"
          ;;
        *)
          echo "[entrypoint] Unknown APP_MODE='${APP_MODE}'"; exit 1;;
      esac
      ;;
    worker)
      echo "[entrypoint] Starting worker..."
      exec python -m jobs.worker
      ;;
    cron)
      echo "[entrypoint] Running scheduled task..."
      exec python -m jobs.daily
      ;;
    *)
      echo "[entrypoint] Unknown SERVICE='${SERVICE}'"; exit 1;;
  esac
}

# --- Main
run_migrations
run_service
