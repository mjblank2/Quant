from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date
from typing import Iterable
from sqlalchemy import text, bindparam
from db import engine, upsert_dataframe, Trade, TargetPosition
from config import PREFERRED_MODEL, STARTING_CAPITAL
from utils.price_utils import select_price_as

import logging
from portfolio.optimizer import build_portfolio
from tax.lots import rebuild_tax_lots_from_trades

log = logging.getLogger(__name__)


def _get_current_shares() -> pd.Series:
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
    try:
        with engine.connect() as con:
            nav = con.execute(
                text("SELECT nav FROM system_state WHERE id = 1")
            ).scalar()
    except Exception:
        nav = None
    return float(nav) if (nav is not None and nav > 0) else STARTING_CAPITAL


def _load_latest_predictions() -> pd.DataFrame:
    with engine.connect() as con:
        preds = pd.read_sql_query(
            text(
                """
            WITH target AS (SELECT MAX(ts) AS mx FROM predictions WHERE model_version = :mv)
            SELECT symbol, ts, y_pred FROM predictions WHERE model_version=:mv AND ts=(SELECT mx FROM target)
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
                SELECT symbol, ts, y_pred FROM predictions WHERE ts=(SELECT MAX(ts) FROM predictions)
            """
                ),
                con,
            )
    return preds


def _load_tca_cols(
    symbols: Iterable[str],
    *,
    attempts: int = 3,
    delay: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch supplementary columns (vol_21, adv_usd_21, size_ln, mom_21,
    turnover_21, beta_63) for the given symbols from the ``features`` table.

    This helper queries the most recent ``ts`` in the ``features`` table and
    returns a DataFrame with one row per symbol.  It includes a simple
    retry loop to mitigate transient database connection errors (e.g.,
    ``psycopg.OperationalError: SSL SYSCALL error: EOF detected``).  By
    default it will attempt the query up to three times, sleeping for
    ``delay`` seconds between attempts.

    Parameters
    ----------
    symbols : Iterable[str]
        List of symbols to fetch supplementary columns for.
    attempts : int, optional
        Number of times to retry the query on connection failure.  Defaults
        to 3.
    delay : float, optional
        Seconds to sleep between retry attempts.  Defaults to 1.0.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ``[symbol, vol_21, adv_usd_21, size_ln,
        mom_21, turnover_21, beta_63]``.  If ``symbols`` is empty or the
        query fails on all attempts, an empty DataFrame with those columns
        is returned.
    """
    symbols = list(dict.fromkeys(symbols))
    # Return an empty DataFrame with the correct schema if there are no symbols.
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

    # Build the query once. Note: bindparams(expanding=True) is required when
    # passing a list/tuple for the ``IN`` clause.
    stmt = text(
        """
        SELECT symbol, ts, vol_21, adv_usd_21, size_ln, mom_21, turnover_21, beta_63
        FROM features WHERE symbol IN :syms AND ts = (SELECT MAX(ts) FROM features)
    """
    ).bindparams(bindparam("syms", expanding=True))

    # Import here to avoid unconditional dependency when not needed.
    try:
        from sqlalchemy.exc import OperationalError
    except ImportError:
        OperationalError = Exception  # fallback

    params = {"syms": tuple(symbols)}
    last_err: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            with engine.connect() as con:
                df = pd.read_sql_query(stmt, con, params=params)
            return df
        except OperationalError as e:
            last_err = e
            # Log the error and retry if more attempts remain
            log.warning(
                "Supplementary data query attempt %s/%s failed: %s",
                attempt,
                attempts,
                e,
            )
            if attempt < attempts:
                # Delay before retrying
                import time

                time.sleep(max(0.0, delay))
                continue
            else:
                break

    # If all attempts failed, log and return an empty DataFrame
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


def generate_today_trades() -> pd.DataFrame:
    log.info("Starting trade generation (v16 optimizer).")
    # Optional: reconstruct tax lots from trades for penalties
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

    all_syms = sorted(
        set(weights.index.tolist()).union(current_shares.index.tolist())
    )
    # arrival prices
    stmt_px = text(
        f"""
        WITH latest AS (SELECT symbol, MAX(ts) ts FROM daily_bars WHERE symbol IN :syms GROUP BY symbol)
        SELECT b.symbol, {select_price_as('px')}
        FROM daily_bars b JOIN latest l ON b.symbol = l.symbol AND b.ts=l.ts
    """
    ).bindparams(bindparam("syms", expanding=True))
    with engine.connect() as con:
        px_df = pd.read_sql_query(
            stmt_px, con, params={"syms": tuple(all_syms)}
        )
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
        tgt_shs = int(np.floor(abs(tgt_notional) / price)) * (
            1 if tgt_notional >= 0 else -1
        )

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

    # Persist
    if target_rows:
        upsert_dataframe(
            pd.DataFrame(target_rows), TargetPosition, ["ts", "symbol"]
        )

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
    generate_today_trades()
