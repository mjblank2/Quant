#!/usr/bin/env bash
set -Eeuo pipefail
printf "[cron_universe] 🕰️ Start: %s\n" "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[cron_universe] SERVICE=${SERVICE:-} PYTHONPATH=${PYTHONPATH:-}"
python -m data.universe
status=$?
printf "[cron_universe] 🕰️ End: %s (exit=%s)\n" "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$status"
exit "$status"