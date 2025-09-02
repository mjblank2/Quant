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

from config import DATABASE_URL  # noqa: E402

# Handle missing DATABASE_URL gracefully during migrations
def get_base_metadata():
    """Import and return Base metadata, handling DATABASE_URL requirements."""
    try:
        from db import Base
        return Base.metadata
    except RuntimeError as e:
        if "DATABASE_URL environment variable is required" in str(e):
            # Create a minimal Base for migrations when DATABASE_URL is not available
            from sqlalchemy.orm import DeclarativeBase
            
            class FallbackBase(DeclarativeBase):
                pass
            
            # Return empty metadata for migrations - this allows Alembic to run
            # but migrations may fail if they need actual database connections
            return FallbackBase.metadata
        else:
            raise

config = context.config
config.set_main_option(
    "sqlalchemy.url",
    DATABASE_URL or "sqlite:///alembic_dummy.db",
)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = get_base_metadata()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
