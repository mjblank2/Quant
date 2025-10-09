#!/usr/bin/env python
"""
Run validation metrics for the 5-day strategy.

This script connects to the project's database to retrieve the daily
returns for the primary trading strategy and computes the Sharpe ratio,
probabilistic Sharpe ratio (PSR), and deflated Sharpe ratio (DSR).
The output is printed to stdout in a human-readable format. It is
intended to be executed manually or as part of a CI step to
benchmark the strategy's statistical significance.
"""

from __future__ import annotations

import argparse
import logging
from typing import List

import pandas as pd
from sqlalchemy import text

from db import engine
from validation.metrics import sharpe_ratio, annualized_sharpe, psr, dsr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_strategy_returns(limit: int = 365) -> pd.Series:
    """Load recent daily returns of the main strategy from the database."""
    query = text(
        """
        SELECT ts, return
        FROM strategy_daily_returns
        WHERE strategy_name = '5_day_model'
        ORDER BY ts DESC
        LIMIT :limit
        """
    )
    with engine.connect() as con:
        df = pd.read_sql_query(query, con, params={"limit": limit}, parse_dates=["ts"])
    if df.empty:
        logger.warning("No strategy returns found in the database.")
        return pd.Series(dtype=float)
    df = df.sort_values("ts")
    return pd.Series(df["return"].values, index=df["ts"])


def main(candidate_srs: List[float] | None) -> None:
    returns = load_strategy_returns()
    if returns.empty:
        print("No returns available for validation.")
        return
    sr = sharpe_ratio(returns)
    sr_ann = annualized_sharpe(returns)
    psr_val = psr(returns, sr_benchmark=0.0)
    dsr_val = dsr(returns, candidate_srs or [])
    print(f"Periodic Sharpe ratio:    {sr:.4f}")
    print(f"Annualized Sharpe ratio:  {sr_ann:.4f}")
    print(f"Probabilistic Sharpe (PSR): {psr_val:.4f}")
    if candidate_srs:
        print(f"Deflated Sharpe (DSR):     {dsr_val:.4f} (based on {len(candidate_srs)} candidates)")
    else:
        print("Deflated Sharpe (DSR) not computed: no candidate SRs provided")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Sharpe-based validation metrics.")
    parser.add_argument(
        "--candidate-srs",
        nargs="*",
        type=float,
        default=None,
        help="List of Sharpe ratios from candidate strategies for DSR computation",
    )
    args = parser.parse_args()
    main(args.candidate_srs)