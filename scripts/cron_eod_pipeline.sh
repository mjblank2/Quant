#!/usr/bin/env bash
set -Eeuo pipefail
printf "[cron_eod] 🕰️ Start: %s\n" "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[cron_eod] SERVICE=${SERVICE:-} PYTHONPATH=${PYTHONPATH:-}"
python run_pipeline.py
status=$?
printf "[cron_eod] 🕰️ End: %s (exit=%s)\n" "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$status"
exit "$status"