from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date
from typing import Iterable
from sqlalchemy import text, bindparam
from db import engine, upsert_dataframe, Trade, TargetPosition
from config import PREFERRED_MODEL, STARTING_CAPITAL

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


def _load_tca_cols(symbols: Iterable[str]) -> pd.DataFrame:
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
    stmt = text(
        """
        SELECT symbol, ts, vol_21, adv_usd_21, size_ln, mom_21, turnover_21, beta_63
        FROM features WHERE symbol IN :syms AND ts = (SELECT MAX(ts) FROM features)
    """
    ).bindparams(bindparam("syms", expanding=True))
    with engine.connect() as con:
        df = pd.read_sql_query(stmt, con, params={"syms": tuple(symbols)})
    return df


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
        """
        WITH latest AS (SELECT symbol, MAX(ts) ts FROM daily_bars WHERE symbol IN :syms GROUP BY symbol)
        SELECT b.symbol, COALESCE(b.adj_close, b.close) AS px
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
