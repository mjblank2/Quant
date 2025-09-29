#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common_cron_preamble.sh"

printf "[cron_eod_pipeline] 🕰️ Start: %s\n" "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[cron_eod_pipeline] SERVICE=${SERVICE:-} PYTHONPATH=${PYTHONPATH:-}"

python run_pipeline.py --yesterday --ignore-market-hours
status=$?

printf "[cron_eod_pipeline] 🕰️ End: %s (exit=%s)\n" "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "${status}"
exit "${status}"
