from __future__ import annotations
import os, sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Ensure project root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

config = context.config

# Normalize DB URL to psycopg v3 driver
from config import _fix_db_url
raw_url = os.getenv("ALEMBIC_URL") or os.getenv("DATABASE_URL") or config.get_main_option("sqlalchemy.url")
config.set_main_option("sqlalchemy.url", _fix_db_url(raw_url) or "sqlite:///alembic_dummy.db")

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Optional metadata import to avoid crashes when DATABASE_URL is not yet set
try:
    from db import Base  # may raise if env misconfigured
    target_metadata = Base.metadata
except Exception:
    Base = None
    target_metadata = None

def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(config.get_section(config.config_ini_section), prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
