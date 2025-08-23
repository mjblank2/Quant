from __future__ import annotations

import os, sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Ensure project root on sys.path so we can import db/config
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from config import _fix_db_url, DATABASE_URL
except Exception:
    # Fallback if config import fails for any reason
    def _fix_db_url(url: str) -> str:
        if not url:
            return url
        # Normalize to psycopg (v3) driver no matter what we get
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        url = url.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)
        return url
    DATABASE_URL = os.getenv("DATABASE_URL", "")

# this is the Alembic Config object, which provides access to the values
# within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import Base metadata lazily (after sys.path patched)
from db import Base  # noqa: E402

target_metadata = Base.metadata

def _resolve_db_url() -> str:
    # Priority: ALEMBIC_URL -> DATABASE_URL (from config.py) -> alembic.ini
    env_url = os.getenv("ALEMBIC_URL") or DATABASE_URL or config.get_main_option("sqlalchemy.url")
    env_url = _fix_db_url(env_url)
    if not env_url:
        env_url = "sqlite:///alembic_dummy.db"
    return env_url

def run_migrations_offline():
    url = _resolve_db_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    url = _resolve_db_url()
    config.set_main_option("sqlalchemy.url", url)
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
