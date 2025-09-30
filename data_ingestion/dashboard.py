from __future__ import annotations
import sys
import os
import logging
from typing import List

import pandas as pd
import sqlalchemy
from sqlalchemy import text, inspect, bindparam

try:
    import streamlit as st
    import plotly.express as px
    IS_STREAMLIT = getattr(getattr(st, "runtime", None), "exists", lambda: False)()
except Exception:
    st = None
    IS_STREAMLIT = False

# Configure logging for the dashboard
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
log = logging.getLogger("dashboard")

# Ensure repo root on sys.path (fallback; packaging the project is preferred)
try:
    if "__file__" in globals():
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
        "Interactive dashboard for trades, positions, predictions and prices (auto‑adapts to your schema)."
    )

    @st.cache_resource
    def get_engine():
        return _initialize_engine()

    engine = get_engine()
else:
    engine = _initialize_engine()

if engine is None and IS_STREAMLIT:
    st.stop()

# ----------------------------------------------------------------------------
# Basic schema init (optional)
# ----------------------------------------------------------------------------
try:
    from db import create_tables  # type: ignore
except Exception:
    def create_tables(*args, **kwargs):
        log.warning("create_tables called but db module not available.")

try:
    if engine:
        # Note: create_tables() relies on db.py's module-level engine; no args.
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
        return inspector.get_table_names(schema="public")
    except Exception as e:
        log.error(f"Failed to list tables: {e}", exc_info=True)
        return []

def table_columns_data(tbl: str) -> List[str]:
    """Return column names for a given table in the public schema."""
    if engine is None:
        return []
    try:
        inspector = inspect(engine)
        columns = inspector.get_columns(tbl, schema="public")
        return [c["name"] for c in columns]
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

def _read_df(sql_query: str, params: dict) -> pd.DataFrame:
    """Execute a SQL query and return a DataFrame."""
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(sql_query), conn, params=params)
    except Exception as e:
        log.error(f"Failed to execute SQL query: {e}", exc_info=True)
        if IS_STREAMLIT and st is not None:
            st.error("An error occurred while fetching data.")
        return pd.DataFrame()

def view_latest_predictions(limit: int = 500) -> pd.DataFrame:
    """Return the latest predictions limited to `limit` rows."""
    if not has_table("predictions"):
        return pd.DataFrame()
    sql_query = """
        SELECT symbol, ts, y_pred, model_version
        FROM predictions
        WHERE ts = (SELECT MAX(ts) FROM predictions)
        ORDER BY y_pred DESC
        LIMIT :lim
    """
    return _read_df(sql_query, {"lim": limit})

# ----------------------------------------------------------------------------
# Additional helpers for enhanced "Top Predictions & Suggested Trades"
# ----------------------------------------------------------------------------
if IS_STREAMLIT:
    @st.cache_data(ttl=300)
    def load_latest_predictions_with_features(n_top: int = 20) -> pd.DataFrame:
        """
        Load the latest predictions for the preferred model and merge with a subset
        of feature columns. Returns the top n rows sorted by predicted return.
        """
        from config import PREFERRED_MODEL  # import here to avoid circular import
        # Grab the most recent predictions for the preferred model
        preds_sql = text("""
            WITH target AS (
                SELECT MAX(ts) AS mx
                FROM predictions
                WHERE model_version = :mv
            )
            SELECT symbol, ts, y_pred
            FROM predictions
            WHERE model_version = :mv
              AND ts = (SELECT mx FROM target)
            ORDER BY y_pred DESC
            LIMIT :limit;
        """)
        with engine.connect() as con:
            preds = pd.read_sql_query(
                preds_sql,
                con,
                params={"mv": PREFERRED_MODEL, "limit": n_top * 5},
            )
        if preds.empty:
            return preds
        # Pull selected feature columns for those symbols from the latest feature date
        syms = preds["symbol"].tolist()
        feats_sql = text("""
            SELECT symbol, ts, vol_21, adv_usd_21, size_ln, mom_21,
                   turnover_21, beta_63
            FROM features
            WHERE symbol IN :syms
              AND ts = (SELECT MAX(ts) FROM features)
        """).bindparams(bindparam("syms", expanding=True))
        with engine.connect() as con:
            feats = pd.read_sql_query(feats_sql, con, params={"syms": tuple(syms)})
        merged = preds.merge(feats, on="symbol", how="left")
        return merged.sort_values("y_pred", ascending=False).head(n_top)

    def compute_display_weights(df: pd.DataFrame) -> pd.Series:
        """Normalise predicted returns to positive weights for display."""
        if df.empty:
            return pd.Series(dtype=float)
        preds = df["y_pred"].astype(float)
        min_pred = preds.min()
        # Shift so the minimum is zero
        shifted = preds - min_pred if min_pred < 0 else preds
        total = shifted.sum()
        return shifted / total if total else pd.Series(0.0, index=df.index)

# ----------------------------------------------------------------------------
# Streamlit rendering
# ----------------------------------------------------------------------------
if IS_STREAMLIT:
    # Latest Predictions table
    st.subheader("Latest Predictions")
    st.dataframe(view_latest_predictions())

    # Current positions with share counts and optional cost basis/PnL
    if has_table("current_positions"):
        st.divider()
        st.subheader("Current Positions")
        try:
            positions_df = _read_df(
                """
                SELECT symbol,
                       shares,
                       cost_basis,
                       shares * (
                         SELECT price FROM daily_bars db
                         WHERE db.symbol = current_positions.symbol
                         ORDER BY db.ts DESC LIMIT 1
                       ) AS market_value,
                       (
                         SELECT price FROM daily_bars db
                         WHERE db.symbol = current_positions.symbol
                         ORDER BY db.ts DESC LIMIT 1
                       ) - cost_basis AS pnl_per_share
                FROM current_positions
                ORDER BY shares DESC
                LIMIT 100
                """,
                {},
            )
        except Exception:
            # Fallback if cost_basis or price columns are missing
            positions_df = _read_df(
                "SELECT symbol, shares FROM current_positions ORDER BY shares DESC LIMIT 100",
                {},
            )
        st.dataframe(positions_df)

    # Recent trades generated by the system (most recent first)
    if has_table("trades"):
        st.divider()
        st.subheader("Recent Trades")
        trades_df = _read_df(
            """
            SELECT id, symbol, side, quantity, price, status, trade_date
            FROM trades
            ORDER BY trade_date DESC, id DESC
            LIMIT 100
            """,
            {},
        )
        st.dataframe(trades_df)

    # Show the latest system net asset value (NAV) if available
    if has_table("system_state"):
        nav_df = _read_df(
            "SELECT nav, ts FROM system_state ORDER BY ts DESC LIMIT 1",
            {},
        )
        if not nav_df.empty:
            nav_val = float(nav_df.iloc[0]["nav"])
            st.divider()
            st.subheader("System NAV")
            st.metric("Net Asset Value", f"${nav_val:,.2f}")

    # --------------------------------------------------------------------------
    # Top Predictions & Suggested Trades (enhanced section for professional users)
    # --------------------------------------------------------------------------
    st.divider()
    st.subheader("Top Predictions & Suggested Trades")
    top_n = st.slider(
        "Select number of top signals to display",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        help="Controls how many high-confidence names are shown.",
    )

    try:
        preds_df = load_latest_predictions_with_features(top_n)
        if preds_df.empty:
            st.info("No predictions available. Run the training pipeline to generate signals.")
        else:
            preds_df["weight"] = compute_display_weights(preds_df)
            # Round numeric columns for readability
            for col in preds_df.select_dtypes(include=["float", "int"]).columns:
                preds_df[col] = preds_df[col].round(4)

            st.dataframe(preds_df.set_index("symbol"), use_container_width=True)

            # Bar chart of predicted returns
            fig_preds = px.bar(
                preds_df,
                x="symbol",
                y="y_pred",
                title="Predicted Returns for Top Signals",
                labels={"y_pred": "Predicted Return", "symbol": "Symbol"},
            )
            st.plotly_chart(fig_preds, use_container_width=True)

            with st.expander("What do these numbers mean?"):
                st.markdown(
                    """
- **Predicted Return (`y_pred`)** – The ensemble model’s forecast of next-period return. Higher values imply stronger expected outperformance.
- **Weight** – A normalised representation of each signal’s relative strength. The actual trading algorithm may apply a more sophisticated optimiser, but this gives a sense of conviction.
- **Volatility (21d)** – Rolling 21-day standard deviation of returns.
- **ADV USD (21d)** – Average dollar volume over 21 days (liquidity proxy).
- **Size (ln MV)** – Natural log of market capitalisation.
- **Momentum (21d)** – Short-term momentum indicator.
- **Turnover (21d)** – Volume relative to shares outstanding.
- **Beta (63d)** – Beta versus the market.
"""
                )

            with st.expander("Model & Process Overview"):
                st.markdown(
                    """
The small‑cap quant system uses an ensemble of machine‑learning models—gradient boosting (XGBoost, LightGBM, CatBoost), random forests, neural networks and a simple reinforcement‑learning‑inspired model—to forecast expected returns. Models are trained on historical price, volume, momentum, volatility and fundamental factors, and blended according to their information coefficient in the current regime. A risk‑aware optimiser then translates signals into portfolio weights. This section helps you see not just the final ranking but also the underlying features, so you understand why each stock appears in the list.
"""
                )
    except Exception as e:
        st.error(f"Failed to load predictions: {e}")

    # --------------------------------------------------------------------------
    # Pipeline Controls: Trigger the full end-of-day pipeline manually
    # This allows a user to run the sequence: ingest data → build features →
    # train models → generate trades, without relying on cron jobs or Celery.
    # --------------------------------------------------------------------------
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

