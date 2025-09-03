"""Alembic migration environment setup."""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Ensure project root on path before importing local modules
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import DATABASE_URL and Base from the application's config and db modules
# This ensures a single source of truth for the database connection and metadata
try:
    from config import DATABASE_URL
    from db import Base
    target_metadata = Base.metadata
except ImportError as e:
    print(f"Error importing application modules: {e}")
    print("Please ensure DATABASE_URL is set in your environment or .env file.")
    # Provide a fallback metadata object for Alembic to function in a limited capacity
    from sqlalchemy.orm import DeclarativeBase
    class FallbackBase(DeclarativeBase):
        pass
    target_metadata = FallbackBase.metadata
    DATABASE_URL = os.getenv("DATABASE_URL", "")
except RuntimeError as e:
     # Handle the case where DATABASE_URL is required but not set in config.py
    print(f"Configuration Error: {e}")
    from sqlalchemy.orm import DeclarativeBase
    class FallbackBase(DeclarativeBase):
        pass
    target_metadata = FallbackBase.metadata
    DATABASE_URL = ""


# this is the Alembic Config object, which provides access to the values
# within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = DATABASE_URL or config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Use the DATABASE_URL from the application's config
    alembic_config = config.get_section(config.config_ini_section)
    alembic_config['sqlalchemy.url'] = DATABASE_URL

    connectable = engine_from_config(
        alembic_config,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

