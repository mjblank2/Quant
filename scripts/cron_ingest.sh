#!/usr/bin/env bash
set -Eeuo pipefail
DAYS_VAL=${DAYS:-7}
printf "[cron_ingest] 🕰️ Start: %s\n" "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[cron_ingest] SERVICE=${SERVICE:-} PYTHONPATH=${PYTHONPATH:-} DAYS=${DAYS_VAL}"
python -m data.ingest --days "${DAYS_VAL}"
status=$?
printf "[cron_ingest] 🕰️ End: %s (exit=%s)\n" "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$status"
exit "$status"