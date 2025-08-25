import os
import pandas as pd
import sqlalchemy
import streamlit as st

st.set_page_config(page_title="Blank Capital Quant - Dashboard", layout="wide")

st.title("Blank Capital Quant")
st.caption("Minimal dashboard placeholder (you can enhance this anytime).")

@st.cache_resource
def get_engine():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        st.warning("DATABASE_URL not set; showing demo content.")
        return None
    # Render often provides postgres://; SQLAlchemy 2 prefers postgresql+psycopg://
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)
    try:
        return sqlalchemy.create_engine(db_url, pool_pre_ping=True)
    except Exception as e:
        st.error(f"Failed to create engine: {e}")
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
        try:
            with engine.connect() as conn:
                # Try trivial query
                result = conn.exec_driver_sql("SELECT 1 AS ok").fetchone()
                st.success(f"DB OK = {result[0]}")
        except Exception as e:
            st.error(f"DB ping failed: {e}")

st.markdown("---")
st.subheader("Demo Table")
df = pd.DataFrame({
    "symbol": ["AAPL", "MSFT", "GOOG"],
    "weight": [0.33, 0.33, 0.34],
})
st.dataframe(df, use_container_width=True)
