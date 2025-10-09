"""
Helper script to synchronize Top-15 portfolio state with the database.

After ``run_daily_update`` writes the current Top-N symbols to the
``top15_portfolio_state.json`` file, this module can be used to update
persistent database tables. It inserts new holdings with today's date,
removes holdings no longer present, and logs buy/sell intents in
``top15_trade_intents``. This is optional, but using it provides a
durable history of positions for the dashboard to read.
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Set

import pandas as pd
from sqlalchemy import text

from db import engine

STATE_PATH = "top15_portfolio_state.json"


def _load_state_syms(path: str = STATE_PATH) -> list[str]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return list(data.get("symbols", []))
    except Exception:
        return []


def sync_state_to_db(state_path: str = STATE_PATH) -> None:
    """Synchronize JSON state file with DB tables.

    This function reads the list of symbols from ``state_path`` and updates
    ``top15_holdings`` and ``top15_trade_intents`` in the database. It
    inserts new entries for symbols not currently held and removes
    positions that are no longer present, logging the corresponding
    buy/sell intents.
    """
    syms = set(_load_state_syms(state_path))
    today = date.today()
    with engine.begin() as con:
        try:
            db_holdings = pd.read_sql_query(text("SELECT symbol, entry_date FROM top15_holdings"), con)
            held: Set[str] = set(db_holdings["symbol"].tolist()) if not db_holdings.empty else set()
        except Exception:
            held = set()
        to_insert = sorted(syms - held)
        to_delete = sorted(held - syms)
        # Inserts
        for s in to_insert:
            con.execute(
                text("INSERT INTO top15_holdings(symbol, entry_date) VALUES (:s, :d) ON CONFLICT (symbol) DO NOTHING"),
                {"s": s, "d": today},
            )
            con.execute(
                text(
                    """
                    INSERT INTO top15_trade_intents(symbol, side, suggested_weight, reason)
                    VALUES (:s, 'BUY', NULL, 'Entered Top-N (sync)')
                    """
                ),
                {"s": s},
            )
        # Deletes
        for s in to_delete:
            con.execute(text("DELETE FROM top15_holdings WHERE symbol = :s"), {"s": s})
            con.execute(
                text(
                    """
                    INSERT INTO top15_trade_intents(symbol, side, suggested_weight, reason)
                    VALUES (:s, 'SELL', NULL, 'Dropped from Top-N (sync)')
                    """
                ),
                {"s": s},
            )