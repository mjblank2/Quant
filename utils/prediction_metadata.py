"""Utility helpers for enriching model prediction batches before persistence."""

from __future__ import annotations

from typing import Any

import pandas as pd


def as_naive_utc(ts: Any) -> pd.Timestamp:
    """Convert arbitrary timestamp-like input into a timezone-naive UTC Timestamp."""
    stamp = pd.Timestamp(ts)
    if pd.isna(stamp):
        raise ValueError("created_at timestamp cannot be NA")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.tz_localize(None)


def with_prediction_metadata(
    df: pd.DataFrame,
    horizon: int,
    created_at: Any,
) -> pd.DataFrame:
    """Attach horizon and created_at columns to prediction batches safely."""
    enriched = df.copy()
    enriched["horizon"] = int(horizon)
    enriched["created_at"] = as_naive_utc(created_at)
    return enriched

