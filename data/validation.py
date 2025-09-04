
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
from sqlalchemy import text

from db import engine  # type: ignore
try:
    from config import ENABLE_DATA_VALIDATION
except Exception:
    ENABLE_DATA_VALIDATION = False

log = logging.getLogger("data.validation")

@dataclass
class ValidationResult:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    anomalies: Optional[pd.DataFrame] = None

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

def detect_price_anomalies(symbol: str | None = None, lookback_days: int = 30) -> ValidationResult:
    result = ValidationResult()
    if not ENABLE_DATA_VALIDATION or engine is None:
        return result

    params: dict = {'lookback_days': lookback_days}
    where_clauses = ["ts >= CURRENT_DATE - (:lookback_days * INTERVAL '1 day')"]
    if symbol:
        where_clauses.append("symbol = :symbol")
        params['symbol'] = symbol
    where_clause = "WHERE " + " AND ".join(where_clauses)

    query = f"""
    SELECT symbol, ts, close,
           LAG(close) OVER (PARTITION BY symbol ORDER BY ts) AS prev_close,
           volume,
           AVG(volume) OVER (PARTITION BY symbol ORDER BY ts ROWS 20 PRECEDING) AS avg_volume
    FROM daily_bars
    {where_clause}
    ORDER BY symbol, ts
    """

    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=params, parse_dates=['ts'])
    except Exception as e:
        log.error(f"Error fetching data for anomaly detection: {e}", exc_info=True)
        result.add_error(f"Database error during anomaly detection: {e}")
        return result

    if df.empty:
        return result

    df['ret'] = df['close'] / df['prev_close'] - 1.0
    # basic anomaly flag: > 5 sigma on volume or > 20% move
    df['vol_sigma'] = (df['volume'] - df['avg_volume']) / (df['avg_volume'].replace(0, pd.NA))
    anomalies = df[(df['ret'].abs() > 0.2) | (df['vol_sigma'].abs() > 5)]
    result.anomalies = anomalies
    return result
