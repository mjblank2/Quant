# =============================================================================
# Module: data_ingestion/dashboard.py
# =============================================================================
import sys
import os
import logging
from datetime import date, timedelta
from typing import List, Optional, Tuple, Dict, Any
import asyncio

import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import text, inspect
from sqlalchemy.engine import Connection
from sqlalchemy import exc
import streamlit as st

# Configure logging globally
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
log = logging.getLogger("dashboard")

# Ensure repo root on sys.path (Fallback; packaging the project is preferred)
try:
    if '__file__' in globals():
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
except Exception as e:
    log.warning(f"Could not modify sys.path: {e}")

# Helper to detect if running in Streamlit context
try:
    IS_STREAMLIT = st.runtime.exists()
except Exception:
    IS_STREAMLIT = False

if IS_STREAMLIT:
    try:
        st.set_page_config(page_title="Blank Capital Quant – Pro Dashboard", layout="wide")
        st.title("Blank Capital Quant")
        st.caption("Interactive dashboard for Trades, Positions, Predictions, and Prices (auto-adapts to your schema).")
    except Exception as e:
        log.warning(f"Failed to initialize Streamlit page config: {e}")

# =============================================================================
# Configuration and Dependencies (Mocked for Monolithic Structure)
# =============================================================================
try:
    from db import upsert_dataframe, DailyBar, DataValidationLog, DataLineage, Universe, Fundamentals, UniverseHistory, AltSignal, RussellMembership, create_tables
    from config import (
        APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, ALPACA_DATA_FEED, 
        TIINGO_API_KEY, POLYGON_API_KEY, ENABLE_DATA_VALIDATION, DEFAULT_KNOWLEDGE_LATENCY_DAYS,
        ENABLE_TIMESCALEDB, TIMESCALEDB_CHUNK_TIME_INTERVAL, MARKET_CAP_MAX, ADV_USD_MIN,
        UNIVERSE_FILTER_RUSSELL, RUSSELL_INDEX, DATA_STALENESS_THRESHOLD_HOURS,
        PRICE_ANOMALY_THRESHOLD_SIGMA, VOLUME_ANOMALY_THRESHOLD_SIGMA
    )
    from utils_http import get_json, get_json_async
except Exception as e:
    log.warning(f"Missing core dependencies (db, config, utils_http). Functionality will be limited. Error: {e}")
    # Minimal placeholders
    DailyBar = DataValidationLog = DataLineage = Universe = Fundamentals = UniverseHistory = AltSignal = RussellMembership = None
    APCA_API_KEY_ID = APCA_API_SECRET_KEY = APCA_API_BASE_URL = ALPACA_DATA_FEED = TIINGO_API_KEY = POLYGON_API_KEY = None
    ENABLE_DATA_VALIDATION = False; DEFAULT_KNOWLEDGE_LATENCY_DAYS = 1; ENABLE_TIMESCALEDB = False
    TIMESCALEDB_CHUNK_TIME_INTERVAL = '1 month'; MARKET_CAP_MAX = 3000000000.0; ADV_USD_MIN = 25000.0
    UNIVERSE_FILTER_RUSSELL = False; RUSSELL_INDEX = None; DATA_STALENESS_THRESHOLD_HOURS = 48
    PRICE_ANOMALY_THRESHOLD_SIGMA = 5.0; VOLUME_ANOMALY_THRESHOLD_SIGMA = 10.0
    def upsert_dataframe(*args, **kwargs): log.warning("upsert_dataframe called but dependencies missing.")
    def get_json(*args, **kwargs): return None
    async def get_json_async(*args, **kwargs): return None
    def create_tables(*args, **kwargs): log.warning("create_tables called but dependencies missing.")

# =============================================================================
# Database Connection Handling
# =============================================================================

def _normalize_dsn(url: str) -> str:
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url

def _initialize_engine():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        log.error("DATABASE_URL environment variable is not set.")
        if IS_STREAMLIT:
            st.error("DATABASE_URL is not set. Set it and refresh.")
        return None
    db_url = _normalize_dsn(db_url)
    try:
        engine = sqlalchemy.create_engine(db_url, pool_pre_ping=True, connect_args={"connect_timeout": 10} if "postgresql" in db_url else {})
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

# Initialize schema
try:
    if engine:
        create_tables(engine)
except Exception as e:
    log.warning(f"Schema init skipped or failed: {e}", exc_info=True)

# =============================================================================
# Dashboard Helper Functions and Views (Summarized)
# =============================================================================

def list_tables_data() -> List[str]:
    if engine is None: return []
    try:
        inspector = inspect(engine)
        try:
            return inspector.get_table_names(schema='public')
        except Exception:
            return inspector.get_table_names()
    except Exception as e:
        log.error(f"Failed to list tables: {e}", exc_info=True)
        return []

def table_columns_data(tbl: str) -> List[str]:
    if engine is None: return []
    try:
        inspector = inspect(engine)
        try:
            columns = inspector.get_columns(tbl, schema='public')
        except Exception:
            columns = inspector.get_columns(tbl)
        return [c['name'] for c in columns]
    except Exception as e:
        log.error(f"Failed to get columns for table {tbl}: {e}", exc_info=True)
        return []

# Apply Streamlit caching if running in Streamlit
if IS_STREAMLIT:
    list_tables = st.cache_data(ttl=120)(list_tables_data)
    table_columns = st.cache_data(ttl=120)(table_columns_data)
else:
    list_tables = list_tables_data
    table_columns = table_columns_data

def has_table(name: str) -> bool:
    return name in list_tables()

def _read_df(sql: str, params: dict) -> pd.DataFrame:
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)
    except Exception as e:
        log.error(f"Failed to execute SQL query: {e}", exc_info=True)
        if IS_STREAMLIT:
            st.error("An error occurred while fetching data.")
        return pd.DataFrame()
