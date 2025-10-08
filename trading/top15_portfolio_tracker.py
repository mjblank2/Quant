"""
top15_portfolio_tracker
=======================

Maintains a rolling Top‑N portfolio of the strongest 5‑day predictions after
liquidity and size filters. Produces human‑readable BUY/SELL "intents" and
persists a simple JSON "state" and a CSV trade log.

- get_top_predictions(...)     -> DataFrame of filtered Top‑N with features
- compute_display_weights(df)  -> normalised weights for display
- update_portfolio(prev, ...)  -> (new_topN_df, trades_intents)
- run_daily_update(...)        -> one‑shot: load state, compute, save state/log

This module does **not** place orders. It focuses on transparency for the
dashboard and auditability for research & monitoring.
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text, bindparam

# Reuse helper functions from existing trading code (pulls predictions/features).
try:
    from trading.generate_trades import _load_latest_predictions, _load_tca_cols  # type: ignore
except Exception as e:
    raise ImportError(
        "Could not import prediction helpers from trading.generate_trades"
    ) from e

# Project DB engine
try:
    from db import engine  # type: ignore
except Exception as e:
    raise ImportError(
        "Database engine unavailable; ensure the Quant project is on sys.path"
    ) from e


# ------------------------------ helpers ------------------------------------- #

def compute_display_weights(df: pd.DataFrame) -> pd.Series:
    """
    Normalise predicted returns to positive weights for display.
    If all values are equal or empty, returns zeros.
    """
    if df is None or df.empty or "y_pred" not in df.columns:
        return pd.Series(dtype=float)
    preds = df["y_pred"].astype(float)
    min_pred = preds.min()
    shifted = preds - min_pred if min_pred < 0 else preds
    total = shifted.sum()
    return shifted / total if total else pd.Series(0.0, index=df.index)


def _get_arrival_prices(symbols: Iterable[str]) -> pd.Series:
    """
    Fetch latest close/arrival price for each symbol (from daily_bars).
    """
    syms = list(dict.fromkeys(symbols))
    if not syms:
        return pd.Series(dtype=float)
    stmt = text(
        """
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
        df = pd.read_sql_query(stmt, con, params={"syms": tuple(syms)})
    return df.set_index("symbol")["px"] if not df.empty else pd.Series(dtype=float)


def get_top_predictions(
    n: int = 15,
    min_adv: float = 1_000_000.0,
    max_market_cap: float = 3_000_000_000.0,
) -> pd.DataFrame:
    """
    Return the top `n` predictions after applying liquidity and size filters.
    Columns: symbol, y_pred, adv_usd_21, size_ln, market_cap
    """
    preds = _load_latest_predictions().sort_values("y_pred", ascending=False)
    if preds.empty:
        return pd.DataFrame(columns=["symbol", "y_pred", "adv_usd_21", "market_cap", "size_ln"])
    sup = _load_tca_cols(preds["symbol"].tolist())
    df = preds.merge(sup, on="symbol", how="left")

    # Compute market cap (if log size is present)
    if "size_ln" in df.columns:
        df["market_cap"] = np.exp(df["size_ln"].astype(float))
    else:
        df["market_cap"] = np.nan

    # Liquidity + size screen
    df = df[(df["adv_usd_21"] >= float(min_adv)) & (df["market_cap"] <= float(max_market_cap))]
    df = df.sort_values("y_pred", ascending=False).head(int(n))
    return df[["symbol", "y_pred", "adv_usd_21", "size_ln", "market_cap"]]


def _diff_portfolios(prev_syms: Set[str], new_df: pd.DataFrame) -> List[dict]:
    """
    Compare previous holdings (symbols only) with new Top‑N DataFrame and return
    a list of {symbol, side, weight}.
    """
    new_syms = [] if new_df.empty else new_df["symbol"].tolist()
    buys = sorted(set(new_syms) - set(prev_syms))
    sells = sorted(set(prev_syms) - set(new_syms))

    weights = compute_display_weights(new_df)
    w_map = {s: float(weights.loc[new_df.index[new_df["symbol"] == s][0]]) if not new_df.empty and s in new_syms else 0.0
             for s in buys + sells}

    trades: List[dict] = []
    for s in buys:
        trades.append({"symbol": s, "side": "BUY", "weight": w_map.get(s, 0.0)})
    for s in sells:
        trades.append({"symbol": s, "side": "SELL", "weight": w_map.get(s, 0.0)})
    return trades


def update_portfolio(
    prev_symbols: Iterable[str],
    n: int = 15,
    min_adv: float = 1_000_000.0,
    max_market_cap: float = 3_000_000_000.0,
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Compute today's Top‑N and diff vs `prev_symbols`.
    Returns (new_topN_df, trades_intents).
    """
    prev_set = set(prev_symbols or [])
    new_df = get_top_predictions(n=n, min_adv=min_adv, max_market_cap=max_market_cap)
    trades = _diff_portfolios(prev_set, new_df)
    return new_df, trades


def _load_state(state_path: str) -> Set[str]:
    if not os.path.exists(state_path):
        return set()
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("symbols", []))
    except Exception:
        return set()


def _save_state(symbols: Iterable[str], state_path: str) -> None:
    data = {"symbols": list(symbols)}
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _append_trades_log(trades: List[dict], log_path: str) -> None:
    """
    Append BUY/SELL intents to a CSV file (date, symbol, side, weight).
    """
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
    """
    One‑shot daily update:
      1) Load previous state
      2) Compute today's Top‑N & diff
      3) Append intents to CSV log
      4) Persist new state
    """
    prev = _load_state(state_path)
    new_df, trades = update_portfolio(prev, n=n, min_adv=min_adv, max_market_cap=max_market_cap)
    _append_trades_log(trades, log_path)
    _save_state(new_df["symbol"].tolist(), state_path)
