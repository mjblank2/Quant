from __future__ import annotations

import sys
import os
import logging
from typing import List
import pandas as pd
import sqlalchemy
from sqlalchemy import text, inspect
from sqlalchemy.engine import Connection

try:
    import streamlit as st
    IS_STREAMLIT = getattr(getattr(st, "runtime", None), "exists", lambda: False)()
except Exception:
    st = None
    IS_STREAMLIT = False

# Configure logging for the dashboard
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
log = logging.getLogger("dashboard")

# Ensure repo root on sys.path (fallback; packaging the project is preferred)
try:
    if '__file__' in globals():
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
except Exception as e:
    log.warning(f"Could not modify sys.path: {e}")

# ----------------------------------------------------------------------------
# DB connection helpers
# ----------------------------------------------------------------------------
def _normalize_dsn(url: str) -> str:
    """Normalize PostgreSQL DSNs for SQLAlchemy compatibility."""
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _initialize_engine():
    """Create a SQLAlchemy engine from the DATABASE_URL environment variable."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        log.error("DATABASE_URL environment variable is not set.")
        if IS_STREAMLIT:
            st.error("DATABASE_URL is not set. Set it and refresh.")
        return None
    db_url = _normalize_dsn(db_url)
    try:
        engine = sqlalchemy.create_engine(
            db_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 10},
        )
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return engine
    except Exception as e:
        log.error(f"Failed to connect to DB: {e}", exc_info=True)
        if IS_STREAMLIT:
            st.error(f"Failed to connect to DB: {e}")
        return None


# Initialize the database engine depending on whether we're running in Streamlit
if IS_STREAMLIT:
    st.set_page_config(page_title="Blank Capital Quant – Pro Dashboard", layout="wide")
    st.title("Blank Capital Quant")
    st.caption(
        "Interactive dashboard for Trades, Positions, Predictions, and Prices (auto-adapts to your schema)."
    )

    @st.cache_resource
    def get_engine():
        return _initialize_engine()

    engine = get_engine()
else:
    engine = _initialize_engine()

if engine is None and IS_STREAMLIT:
    # Stop Streamlit execution if we couldn't initialize the engine
    st.stop()


# ----------------------------------------------------------------------------
# Basic schema init (optional)
# ----------------------------------------------------------------------------
try:
    # Import create_tables from our db module. This helper creates tables defined
    # via SQLAlchemy's Base metadata. It takes no parameters in the current
    # implementation, so we shouldn't pass the engine to it.
    from db import create_tables  # type: ignore
except Exception:
    def create_tables(*args, **kwargs):
        log.warning("create_tables called but db module not available.")

try:
    if engine:
        # Note: create_tables() does not accept parameters. Passing an engine
        # would raise a TypeError. It relies on db.py's module-level engine.
        create_tables()
except Exception as e:
    log.warning(f"Schema init skipped or failed: {e}", exc_info=True)


# ----------------------------------------------------------------------------
# Dashboard helpers
# ----------------------------------------------------------------------------
def list_tables_data() -> List[str]:
    """Return a list of table names in the public schema."""
    if engine is None:
        return []
    try:
        inspector = inspect(engine)
        return inspector.get_table_names(schema='public')
    except Exception as e:
        log.error(f"Failed to list tables: {e}", exc_info=True)
        return []


def table_columns_data(tbl: str) -> List[str]:
    """Return column names for a given table in the public schema."""
    if engine is None:
        return []
    try:
        inspector = inspect(engine)
        columns = inspector.get_columns(tbl, schema='public')
        return [c['name'] for c in columns]
    except Exception as e:
        log.error(f"Failed to get columns for table {tbl}: {e}", exc_info=True)
        return []


# Use Streamlit caching to avoid repeated database lookups when running in Streamlit
if IS_STREAMLIT:
    list_tables = st.cache_data(ttl=120)(list_tables_data)
    table_columns = st.cache_data(ttl=120)(table_columns_data)
else:
    list_tables = list_tables_data
    table_columns = table_columns_data


def has_table(name: str) -> bool:
    """Check whether a table exists in the public schema."""
    return name in list_tables()


def _read_df(sql: str, params: dict) -> pd.DataFrame:
    """Execute a SQL query and return a DataFrame."""
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)
    except Exception as e:
        log.error(f"Failed to execute SQL query: {e}", exc_info=True)
        if IS_STREAMLIT and st is not None:
            st.error("An error occurred while fetching data.")
        return pd.DataFrame()


# Minimal example views (customize per your schema)
def view_latest_predictions(limit: int = 500) -> pd.DataFrame:
    """Return the latest predictions limited to `limit` rows."""
    if not has_table("predictions"):
        return pd.DataFrame()
    sql = '''
        SELECT symbol, ts, y_pred, model_version
        FROM predictions
        WHERE ts = (SELECT MAX(ts) FROM predictions)
        ORDER BY y_pred DESC
        LIMIT :lim
    '''
    return _read_df(sql, {"lim": limit})


if IS_STREAMLIT:
    # Show the predictions table in the dashboard
    st.subheader("Latest Predictions")
    st.dataframe(view_latest_predictions())

    # ----------------------------------------------------------------------
    # Pipeline Controls: Trigger the full end-of-day pipeline manually
    # This allows a user to run the sequence: ingest data → build features →
    # train models → generate trades, without relying on cron jobs or Celery.
    # ----------------------------------------------------------------------
    try:
        from run_pipeline import main as run_full_pipeline_main  # type: ignore
    except Exception:
        run_full_pipeline_main = None

    st.divider()
    st.subheader("Pipeline Controls")
    run_btn = st.button(" Run End‑of‑Day Pipeline")
    if run_btn:
        if run_full_pipeline_main is None:
            st.error("Pipeline module is unavailable. Ensure run_pipeline.py exists in your project.")
        else:
            with st.spinner("Running end‑of‑day pipeline…"):
                try:
                    success = run_full_pipeline_main(sync_broker=False)
                    if success:
                        st.success("Pipeline completed successfully.")
                    else:
                        st.error("Pipeline completed with errors. Check logs for details.")
                except Exception as e:
                    st.error(f"Pipeline execution failed: {e}")
