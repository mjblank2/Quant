"""
top15_portfolio_tracker
=======================

This module implements a simple helper to maintain a rolling portfolio of
the top‑N predictions produced by the existing small‑cap quant system.  It
filters the universe of signals to exclude hard‑to‑trade names, applies
liquidity and market‑capitalisation thresholds, and then normalises the
remaining predictions into positive weights.  A lightweight state file
tracks the current holdings and outputs a list of trades (buys and sells)
whenever the composition of the top‑N names changes.

The intent of this script is to give analysts a transparent view of what
would happen if they committed to always holding the top N names according
to the machine learning forecasts.  It does **not** place any orders or
sync with a broker; rather, it writes a CSV log so performance can be
monitored over time.  To integrate this into a production pipeline you
could call ``update_portfolio`` from an orchestrator each day after new
predictions have been generated.

Usage example::

    from trading.top15_portfolio_tracker import update_portfolio

    # load last state from disk (or start empty if none exists)
    prev_symbols = ...
    new_portfolio_df, trades = update_portfolio(prev_symbols,
                                                min_adv=1_000_000,
                                                max_market_cap=3e9,
                                                top_n=15)
    # write ``trades`` to your logging table and persist ``new_portfolio_df``

"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text, bindparam

# Reuse helper functions from existing trading code.  These private
# functions pull the latest predictions and supplementary feature
# columns from the database.  We import them here rather than
# duplicating SQL logic.
try:
    from trading.generate_trades import _load_latest_predictions, _load_tca_cols  # type: ignore
except Exception as e:
    raise ImportError(
        "Could not import prediction helpers from trading.generate_trades"
    ) from e

# Import the database engine so we can fetch arrival prices when sizing trades.
try:
    from db import engine  # type: ignore
except Exception as e:
    # If db import fails, raise a helpful error.  In development you can
    # adjust sys.path or install project dependencies so that ``db`` is
    # discoverable.
    raise ImportError(
        "Database engine unavailable; ensure the Quant project is on sys.path"
    ) from e

def compute_display_weights(df: pd.DataFrame) -> pd.Series:
    """Normalise predicted returns to positive weights for display.

    The raw predictions can be negative; this function shifts the
    distribution so the minimum value is zero and then normalises to
    produce a set of weights that sum to 1.  If all values are equal
    (or the sum of shifted values is zero), it returns a zero‑filled
    series.
    """
    if df.empty:
        return pd.Series(dtype=float)
    preds = df["y_pred"].astype(float)
    min_pred = preds.min()
    shifted = preds - min_pred if min_pred < 0 else preds
    total = shifted.sum()
    return shifted / total if total else pd.Series(0.0, index=df.index)


def _get_arrival_prices(symbols: Iterable[str]) -> pd.Series:
    """Fetch latest arrival prices for a set of symbols.

    This helper queries the ``daily_bars`` table to retrieve the most
    recent closing price for each symbol.  It returns a series indexed
    by symbol.  If a symbol lacks a price, that entry will be ``NaN``.
    """
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        return pd.Series(dtype=float)
    stmt = text(
        f"""
        WITH latest AS (
            SELECT symbol, MAX(ts) AS mx
            FROM daily_bars
            WHERE symbol IN :syms
            GROUP BY symbol
        )
        SELECT b.symbol, b.close AS px
        FROM daily_bars AS b
        JOIN latest AS l ON b.symbol = l.symbol AND b.ts = l.mx
    """
    ).bindparams(bindparam("syms", expanding=True))
    with engine.connect() as con:
        df = pd.read_sql_query(stmt, con, params={"syms": tuple(symbols)})
    return df.set_index("symbol")["px"] if not df.empty else pd.Series(dtype=float)


def get_top_predictions(
    n: int = 15,
    min_adv: float = 1_000_000.0,
    max_market_cap: float = 3_000_000_000.0,
) -> pd.DataFrame:
    """Return the top ``n`` predictions after applying liquidity and size filters.

    This function loads the latest model predictions and merges in
    transaction cost analytics (TCA) columns such as the 21‑day average
    dollar volume (``adv_usd_21``) and log market capitalization
    (``size_ln``).  It then computes the actual market cap (by
    exponentiating ``size_ln``), filters out names with insufficient
    liquidity or too large size, and finally returns the ``n`` rows
    with the highest predicted returns.

    Parameters
    ----------
    n: int
        Number of names to select (default 15).
    min_adv: float
        Minimum 21‑day average daily dollar volume.  Names with
        ``adv_usd_21`` below this threshold are considered too illiquid
        and are excluded.
    max_market_cap: float
        Maximum market capitalization allowed.  Only stocks with
        market cap less than or equal to this value will be retained.

    Returns
    -------
    pandas.DataFrame
        Dataframe of the top predictions after filtering, sorted by
        ``y_pred`` descending.  Columns include ``symbol``, ``y_pred``,
        ``adv_usd_21``, ``market_cap`` and the raw ``size_ln``.
    """
    preds = _load_latest_predictions().sort_values("y_pred", ascending=False)
    if preds.empty:
        return pd.DataFrame(
            columns=["symbol", "y_pred", "adv_usd_21", "market_cap", "size_ln"]
        )
    sup = _load_tca_cols(preds["symbol"].tolist())
    df = preds.merge(sup, on="symbol", how="left")
    # Compute actual market cap from log size if available
    if "size_ln" in df.columns:
        df["market_cap"] = np.exp(df["size_ln"].astype(float))
    else:
        df["market_cap"] = np.nan
    # Apply liquidity and size filters
    df = df[(df["adv_usd_21"] >= min_adv) & (df["market_cap"] <= max_market_cap)]
    # Sort and take top N
    df = df.sort_values("y_pred", ascending=False).head(n)
    return df[["symbol", "y_pred", "adv_usd_21", "market_cap", "size_ln"]]


def generate_top_portfolio(
    n: int = 15,
    min_adv: float = 1_000_000.0,
    max_market_cap: float = 3_000_000_000.0,
) -> pd.DataFrame:
    """Return a dataframe of the top predictions with normalised weights.

    This convenience wrapper calls ``get_top_predictions`` to apply
    the liquidity and size filters, then computes positive weights
    using ``compute_display_weights``.  The resulting dataframe
    contains ``symbol``, ``y_pred``, ``adv_usd_21``, ``market_cap`` and
    ``weight`` columns.
    """
    df = get_top_predictions(n=n, min_adv=min_adv, max_market_cap=max_market_cap)
    if df.empty:
        return df.assign(weight=pd.Series(dtype=float))
    weights = compute_display_weights(df)
    df = df.set_index("symbol")
    df["weight"] = weights
    return df.reset_index()


def update_portfolio(
    previous_symbols: Iterable[str],
    n: int = 15,
    min_adv: float = 1_000_000.0,
    max_market_cap: float = 3_000_000_000.0,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Determine new holdings and required trades for a top‑N strategy.

    Given a list or set of ``previous_symbols`` representing the current
    holdings, this function returns the new portfolio dataframe and a
    list of trades needed to transition from the current holdings to
    the new top‑N portfolio.  Trades are represented as dictionaries
    with keys ``symbol``, ``side`` and ``weight``.

    Parameters
    ----------
    previous_symbols: Iterable[str]
        The current set of symbols held in the portfolio.
    n: int
        Number of names to include in the new portfolio (default 15).
    min_adv: float
        Minimum liquidity threshold for ``adv_usd_21``.
    max_market_cap: float
        Maximum market cap threshold.  Only names below this cap are
        considered.

    Returns
    -------
    tuple of (new_portfolio_df, trades_list)
        ``new_portfolio_df`` – DataFrame with columns
        ``symbol``, ``y_pred``, ``adv_usd_21``, ``market_cap`` and
        ``weight`` for the new top‑N names.
        ``trades_list`` – list of dictionaries describing the trades
        required to migrate from ``previous_symbols`` to the new set.
        Each dict contains ``symbol``, ``side`` ("BUY" or "SELL") and
        ``weight`` (the target weight for buys, 0 for sells).
    """
    prev_set: Set[str] = set(previous_symbols)
    new_df = generate_top_portfolio(n=n, min_adv=min_adv, max_market_cap=max_market_cap)
    new_set: Set[str] = set(new_df["symbol"].tolist())
    trades: List[dict] = []
    # Determine buys (names in new but not in previous) and sells (names dropped)
    buys = new_set - prev_set
    sells = prev_set - new_set
    # Map weights for new names
    weight_map = {
        row["symbol"]: float(row["weight"])
        for _, row in new_df.iterrows()
    }
    for sym in buys:
        trades.append({"symbol": sym, "side": "BUY", "weight": weight_map.get(sym, 0.0)})
    for sym in sells:
        trades.append({"symbol": sym, "side": "SELL", "weight": 0.0})
    return new_df, trades


def _load_state(state_path: str) -> Set[str]:
    """Load the previous portfolio symbols from a JSON state file.

    The state file stores a simple JSON object with a single key
    ``symbols`` mapping to a list of ticker strings.  If the file
    cannot be read or does not exist, an empty set is returned.
    """
    if not os.path.exists(state_path):
        return set()
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        syms = data.get("symbols", [])
        return set(syms)
    except Exception:
        return set()


def _save_state(symbols: Iterable[str], state_path: str) -> None:
    """Persist the current portfolio symbols to a JSON state file."""
    data = {"symbols": list(symbols)}
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _append_trades_log(trades: List[dict], log_path: str) -> None:
    """Append trade records to a CSV file for performance monitoring."""
    if not trades:
        return
    import csv
    today = date.today().isoformat()
    header = ["date", "symbol", "side", "weight"]
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        for tr in trades:
            writer.writerow({"date": today, **tr})


def run_daily_update(
    state_path: str = "top15_portfolio_state.json",
    log_path: str = "top15_trades_log.csv",
    n: int = 15,
    min_adv: float = 1_000_000.0,
    max_market_cap: float = 3_000_000_000.0,
) -> None:
    """High level convenience wrapper to update the top‑N portfolio daily.

    This function encapsulates loading the previous state, computing the new
    portfolio and trades, logging any differences and saving the new state.
    When invoked regularly (e.g., via a cron job), it keeps the
    ``top15_trades_log.csv`` up‑to‑date with buy/sell signals and
    maintains a JSON record of the current holdings.
    """
    prev = _load_state(state_path)
    new_df, trades = update_portfolio(prev, n=n, min_adv=min_adv, max_market_cap=max_market_cap)
    # Persist trades to log and update state
    _append_trades_log(trades, log_path)
    _save_state(new_df["symbol"].tolist(), state_path)

    # Optional: return objects for inspection
    return None


if __name__ == "__main__":
    # When run as a script, perform the daily update using default parameters.
    run_daily_update()
