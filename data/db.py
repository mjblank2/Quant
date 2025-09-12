from __future__ import annotations

# Shim to support relative imports inside data/ modules that expect `.db`
# The actual SQLAlchemy models and session live at the repository root `db.py`.
from db import Universe, SessionLocal  # re-export for data.universe and others  # noqa: F401
