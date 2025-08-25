import os
import pandas as pd
import sqlalchemy
import streamlit as st

st.set_page_config(page_title="Blank Capital Quant - Dashboard", layout="wide")

st.title("Blank Capital Quant")
st.caption("Minimal dashboard with working PostgreSQL connection (psycopg v3).")

def _normalize_dsn(url: str) -> str:
    # Accept postgres:// or postgresql:// and upgrade to SQLAlchemy 2 driver string
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url

@st.cache_resource
def get_engine():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        st.warning("DATABASE_URL not set; showing demo content.")
        return None
    db_url = _normalize_dsn(db_url)
    try:
        engine = sqlalchemy.create_engine(db_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return engine
    except Exception as e:
        st.error(f"Failed to create engine: {e}")
        st.code(db_url, language="text")
        return None

engine = get_engine()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Health")
    st.json({
        "env": os.environ.get("RENDER_SERVICE_NAME", "local/dev"),
        "service": os.environ.get("SERVICE", "web"),
        "app_mode": os.environ.get("APP_MODE", "streamlit"),
    })

with col2:
    st.subheader("Database Ping")
    if engine is None:
        st.info("No database connected.")
    else:
        st.success("Connected to database.")

st.markdown("---")
st.subheader("Demo Table")
df = pd.DataFrame({
    "symbol": ["AAPL", "MSFT", "GOOG"],
    "weight": [0.33, 0.33, 0.34],
})
st.dataframe(df, use_container_width=True)
