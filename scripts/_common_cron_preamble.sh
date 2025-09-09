#!/usr/bin/env bash
# Common environment normalization for cron scripts to run identically
# under Docker (/app) and Render native (/opt/render/project/src) runtimes.
#
# Intentionally do NOT set -e/-u here; the calling cron script should manage
# its own safety flags and error handling.

# Normalize PYTHONPATH
if [[ -z "${PYTHONPATH:-}" ]]; then
  if [[ -d "/app" ]]; then
    export PYTHONPATH="/app"
  elif [[ -d "/opt/render/project/src" ]]; then
    export PYTHONPATH="/opt/render/project/src"
  fi
fi

# Best-effort Alembic migrations. Use empty PYTHONPATH so the Alembic CLI
# resolves the application package via env.py (which restores project path).
if command -v alembic >/dev/null 2>&1; then
  if PYTHONPATH="" alembic upgrade heads; then
    echo "[cron] ✅ Alembic upgrade succeeded"
  else
    echo "[cron] ⚠️ Alembic upgrade failed (continuing)"
  fi
else
  echo "[cron] ℹ️ alembic not found; skipping migrations"
fi
