# =============================================================================
# Module: data/validation.py
# =============================================================================
import logging
from typing import List, Optional
import pandas as pd
from sqlalchemy import text

log_validation = logging.getLogger("data.validation")

try:
    from config import ENABLE_DATA_VALIDATION
except Exception:
    ENABLE_DATA_VALIDATION = False

try:
    from data_ingestion.dashboard import engine
except Exception:
    engine = None

class ValidationResult:
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.findings: pd.DataFrame | None = None
    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

def detect_price_anomalies(symbol: Optional[str] = None, lookback_days: int = 30) -> ValidationResult:
    result = ValidationResult()
    if not ENABLE_DATA_VALIDATION or engine is None:
        return result

    params = {'lookback_days': lookback_days}
    where_clauses = ["ts >= CURRENT_DATE - (:lookback_days * INTERVAL '1 day')"]

    if symbol:
        where_clauses.append("symbol = :symbol")
        params['symbol'] = symbol

    where_clause = "WHERE " + " AND ".join(where_clauses)

    query = f"""
    SELECT symbol, ts, close, 
           LAG(close) OVER (PARTITION BY symbol ORDER BY ts) as prev_close,
           volume,
           AVG(volume) OVER (PARTITION BY symbol ORDER BY ts ROWS 20 PRECEDING) as avg_volume
    FROM daily_bars 
    {where_clause}
    ORDER BY symbol, ts
    """

    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=params, parse_dates=['ts'])
        result.findings = df
    except Exception as e:
        log_validation.error(f"Error fetching data for anomaly detection: {e}", exc_info=True)
        result.add_error(f"Database error during anomaly detection: {e}")

    return result
