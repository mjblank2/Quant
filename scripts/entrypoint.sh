#!/usr/bin/env bash
set -euo pipefail

# Ensure we run from the project root (where alembic.ini lives)
cd /app

echo "=== Applying DB migrations ==="
# Use the CLI, not 'python -m alembic'
alembic -c alembic.ini upgrade head

echo "=== Starting Streamlit ==="
# If PORT is not set, default to 8000 (Render sets PORT automatically)
exec streamlit run app.py --server.port "${PORT:-8000}" --server.address 0.0.0.0
