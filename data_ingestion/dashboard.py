from __future__ import annotations
import sys, os, logging
from typing import List
import pandas as pd
import sqlalchemy
from sqlalchemy import text, inspect
import streamlit as st

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
log = logging.getLogger("dashboard")

try:
    if '__file__' in globals():
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
except Exception as e:
    log.warning(f"Could not modify sys.path: {e}")

IS_STREAMLIT = st.runtime.exists()

if IS_STREAMLIT:
    st.set_page_config(page_title="Blank Capital Quant – Pro Dashboard", layout="wide")
    st.title("Blank Capital Quant")
    st.caption("Interactive dashboard for Trades, Positions, Predictions, and Prices.")

def _normalize_dsn(url: str) -> str:
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url

def _initialize_engine():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        if IS_STREAMLIT:
            st.error("DATABASE_URL is not set.")
        return None
    db_url = _normalize_dsn(db_url)
    try:
        engine = sqlalchemy.create_engine(db_url, pool_pre_ping=True, connect_args={"connect_timeout": 10})
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return engine
    except Exception as e:
        log.error(f"Failed to connect to DB: {e}", exc_info=True)
        if IS_STREAMLIT:
            st.error(f"Failed to connect to DB: {e}")
        return None

if IS_STREAMLIT:
    @st.cache_resource
    def get_engine():
        return _initialize_engine()
    engine = get_engine()
else:
    engine = _initialize_engine()

if engine is None and IS_STREAMLIT:
    st.stop()

def list_tables_data() -> List[str]:
    if engine is None: return []
    try:
        inspector = inspect(engine)
        return inspector.get_table_names(schema='public')
    except Exception as e:
        log.error(f"Failed to list tables: {e}", exc_info=True)
        return []

def table_columns_data(tbl: str) -> List[str]:
    if engine is None: return []
    try:
        inspector = inspect(engine)
        columns = inspector.get_columns(tbl, schema='public')
        return [c['name'] for c in columns]
    except Exception as e:
        log.error(f"Failed to get columns for table {tbl}: {e}", exc_info=True)
        return []

if IS_STREAMLIT:
    list_tables = st.cache_data(ttl=120)(list_tables_data)
    table_columns = st.cache_data(ttl=120)(table_columns_data)
else:
    list_tables = list_tables_data
    table_columns = table_columns_data

def has_table(name: str) -> bool:
    return name in list_tables()
