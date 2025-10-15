from __future__ import annotations

"""
Trade generation for the Small‑Cap Quant system.

This module builds on top of the existing v16 trade generator but adds the
ability to synchronise current holdings with Interactive Brokers (IB) via
`ib_insync`.  By default, it still relies on the local ``current_positions``
table, but callers can pass ``portfolio aligned=True`` to refresh that table from a
live IB TWS session before computing trade deltas.

Functions:

* ``generate_today_trades(aligned with targets: bool = False) -> pd.DataFrame``
    Generate today's trades based on the latest predictions and current
    portfolio.  Optionally synchronises positions from IB.

Helper functions (prefixed with ``_``) are internal and mirror the original
implementation for loading predictions, supplementary columns, and system
state.
"""

import logging
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import text, bindparam

from db import engine, upsert_dataframe, Trade, TargetPosition  # type: ignore
from config import PREFERRED_MODEL, STARTING_CAPITAL  # type: ignore
from utils.price_utils import select_price_as  # type: ignore
from portfolio.optimizer import build_portfolio  # type: ignore
from tax.lots import rebuild_tax_lots_from_trades  # type: ignore

log = logging.getLogger(__name__)



def _get_current_shares() -> pd.Series:
    """Load current positions from the database as a Series mapping symbol to shares."""
    try:
        with engine.connect() as con:
            df = pd.read_sql_query(
                text("SELECT symbol, shares FROM current_positions"), con
            )
    except Exception as e:
        log.warning(f"Could not load current positions: {e}")
        return pd.Series(dtype=int)
    return (
        df.set_index("symbol")["shares"]
        if not df.empty
        else pd.Series(dtype=int)
    )



def _get_system_nav() -> float:
    """Return the system's net asset value (NAV) from the database or starting capital."""
    try:
        with engine.connect() as con:
            nav = con.execute(
                text("SELECT nav FROM system_state WHERE id = 1")
            ).scalar()
    except Exception:
        nav = None
    return float(nav) if (nav is not None and nav > 0) else STARTING_CAPITAL



def _load_latest_predictions() -> pd.DataFrame:
    """Load the most recent predictions from the ``predictions`` table."""
    with engine.connect() as con:
        preds = pd.read_sql_query(
            text(
                """
                WITH target AS (SELECT MAX(ts) AS mx FROM predictions WHERE model_version = :mv)
                SELECT symbol, ts, y_pred FROM predictions
                WHERE model_version = :mv AND ts = (SELECT mx FROM target)
                """
            ),
            con,
            params={"mv": PREFERRED_MODEL},
        )
        if preds.empty:
            log.warning(
                f"No predictions for preferred model {PREFERRED_MODEL}. Falling back to latest ts."
            )
            preds = pd.read_sql_query(
                text(
                    """
                    SELECT symbol, ts, y_pred
                    FROM predictions
                    WHERE ts = (SELECT MAX(ts) FROM predictions)
                    """
                ),
                con,
            )
    return preds



def _load_tca_cols(
    symbols: Iterable[str], *, attempts: int = 3, delay: float = 1.0
) -> pd.DataFrame:
    """
    Fetch supplementary columns (vol_21, adv_usd_21, size_ln, mom_21,
    turnover_21, beta_63) for the given symbols from the ``features`` table.

    Includes a simple retry loop to mitigate transient database connection errors.
    Returns an empty DataFrame with the expected columns if the query fails.
    """
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        return pd.DataFrame(
            columns=[
                "symbol",
                "vol_21",
                "adv_usd_21",
                "size_ln",
                "mom_21",
                "turnover_21",
                "beta_63",
            ]
        )
    stmt = (
        text(
            """
            SELECT symbol, ts, vol_21, adv_usd_21, size_ln, mom_21, turnover_21, beta_63
            FROM features
            WHERE symbol IN :syms AND ts = (SELECT MAX(ts) FROM features)
            """
        ).bindparams(bindparam("syms", expanding=True))
    )
    try:
        from sqlalchemy.exc import OperationalError  # type: ignore
    except ImportError:
        OperationalError = Exception  # type: ignore
    params = {"syms": tuple(symbols)}
    last_err: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            with engine.connect() as con:
                df = pd.read_sql_query(stmt, con, params=params)
            return df
        except OperationalError as e:
            last_err = e
            log.warning(
                "Supplementary data query attempt %s/%s failed: %s",
                attempt,
                attempts,
                e,
            )
            if attempt < attempts:
                import time

                time.sleep(max(0.0, delay))
                continue
            else:
                break
    log.error(
        "Failed to fetch supplementary TCA columns after %s attempts: %s",
        attempts,
        last_err,
    )
    return pd.DataFrame(
        columns=[
            "symbol",
            "vol_21",
            "adv_usd_21",
            "size_ln",
            "mom_21",
            "turnover_21",
            "beta_63",
        ]
    )



def generate_today_trades(sync_ib: bool = False) -> pd.DataFrame:
    """
    Generate today's trades based on the latest model predictions and current portfolio.

    This function rebuilds tax lots, loads predictions and supplementary factors,
    computes target weights using the portfolio optimizer, and then diffs against
    the current holdings to produce BUY/SELL orders.  If ``sync_ib`` is True,
    the ``current_positions`` table is refreshed from Interactive Brokers via
    ``trading.ib_connector.update_current_positions_from_ib`` before any
    calculations.

    Parameters
    ----------
    sync_ib : bool, optional
        Whether to synchronise positions from IB before generating trades.
        Defaults to ``False``.

    Returns
    -------
    pandas.DataFrame
        DataFrame of generated trades with columns ``id``, ``symbol``, ``side``,
        ``quantity``, ``price``, ``status``, and ``trade_date``.  If no trades
        are required, the DataFrame will be empty.
    """
    log.info("Starting trade generation (v16 optimizer).")
    # Synchronise portfolio from Interactive Brokers if requested
    if sync_ib:
        try:
            from trading.ib_connector import update_current_positions_from_ib  # type: ignore
            if update_current_positions_from_ib():
                log.info("Successfully synchronised portfolio from Interactive Brokers.")
            else:
                log.warning("IB portfolio synchronisation failed. Proceeding with local positions.")
        except Exception as e:
            log.warning(f"IB portfolio synchronisation exception: {e}")

    # Optional: rebuild tax lots to ensure penalties reflect realised trades
    try:
        rebuild_tax_lots_from_trades()
    except Exception as e:
        log.info(f"Tax lots rebuild skipped/failed: {e}")

    preds = _load_latest_predictions().sort_values("y_pred", ascending=False)
    if preds.empty:
        log.error("No predictions available. Cannot generate trades.")
        return pd.DataFrame(
            columns=[
                "id",
                "symbol",
                "side",
                "quantity",
                "price",
                "status",
                "trade_date",
            ]
        )
    # Merge supplementary columns needed by optimizer
    sup = _load_tca_cols(preds["symbol"].tolist())
    pred_df = preds.merge(sup, on=["symbol"], how="left")
    today = date.today()
    current_shares = _get_current_shares()
    current_nav = _get_system_nav()
    weights = build_portfolio(
        pred_df, today, current_symbols=current_shares.index.tolist()
    )
    all_syms = sorted(set(weights.index.tolist()).union(current_shares.index.tolist()))
    # arrival prices
    stmt_px = (
        text(
            f"""
            WITH latest AS (SELECT symbol, MAX(ts) ts FROM daily_bars WHERE symbol IN :syms GROUP BY symbol)
            SELECT b.symbol, {select_price_as('px')}
            FROM daily_bars b JOIN latest l ON b.symbol = l.symbol AND b.ts = l.ts
            """
        ).bindparams(bindparam("syms", expanding=True))
    )
    with engine.connect() as con:
        px_df = pd.read_sql_query(stmt_px, con, params={"syms": tuple(all_syms)})
    prices = (
        px_df.set_index("symbol")["px"]
        if not px_df.empty
        else pd.Series(dtype=float)
    )
    # Build target positions and trades
    target_rows = []
    trade_rows = []
    for s in all_syms:
        tgt_w = float(weights.get(s, 0.0))
        cur_shs = int(current_shares.get(s, 0))
        price = float(prices.get(s, np.nan))
        if not (np.isfinite(price) and price > 0):
            continue
        tgt_notional = tgt_w * current_nav
        tgt_shs = int(np.floor(abs(tgt_notional) / price)) * (1 if tgt_notional >= 0 else -1)
        # record target (even zero -> intent to close)
        if (tgt_shs != 0) or (cur_shs != 0):
            target_rows.append(
                {
                    "ts": today,
                    "symbol": s,
                    "weight": tgt_w,
                    "price": price,
                    "target_shares": tgt_shs,
                }
            )
        delta = tgt_shs - cur_shs
        if delta == 0:
            continue
        side = "BUY" if delta > 0 else "SELL"
        trade_rows.append(
            {
                "symbol": s,
                "side": side,
                "quantity": abs(delta),
                "price": price,
                "trade_date": today,
                "status": "generated",
            }
        )
    # Persist target positions
    if target_rows:
        upsert_dataframe(pd.DataFrame(target_rows), TargetPosition, ["ts", "symbol"])
    # Return empty DataFrame if no trades
    if not trade_rows:
        log.info("No trades generated (portfolio aligned with targets).")
        return pd.DataFrame(
            columns=[
                "id",
                "symbol",
                "side",
                "quantity",
                "price",
                "status",
                "trade_date",
            ]
        )
    trades_df = pd.DataFrame(trade_rows)
    # Insert trades into database
    with engine.begin() as conn:
        try:
            result = conn.execute(
                Trade.__table__.insert().returning(Trade.id),
                trades_df.to_dict(orient="records"),
            )
            ids = [row[0] for row in result]
            trades_df["id"] = ids
        except Exception as e:
            log.error(f"Failed to insert trades: {e}")
            trades_df["id"] = None
    log.info(f"Generated {len(trades_df)} trades.")
    return trades_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # When run as a script, synchronise positions from IB before generating trades
    generate_today_trades(sync_ib=True)
