from __future__ import annotations

"""
Interactive Brokers (IB) integration utilities.

This module provides helper functions to connect to IB Trader Workstation (TWS),
fetch current portfolio positions, update the local database, and convert
generated trades into IB order objects.  It relies on the `ib_insync` library
to simplify interactions with IB's API.  If `ib_insync` is not installed,
importing this module will not raise, but any IB‑related functions will log
errors when called.

The assistant will never automatically submit orders.  The `submit_ib_orders`
function is provided for completeness but should be invoked only after
explicit human confirmation.
"""

import logging
from typing import Dict, Optional

import pandas as pd

try:
    # ib_insync wraps IB API and simplifies asynchronous calls.
    from ib_insync import IB, Stock, MarketOrder, LimitOrder  # type: ignore
except Exception:
    # Fallback stubs if ib_insync is missing; functions will log errors.
    IB = None  # type: ignore
    Stock = None  # type: ignore
    MarketOrder = None  # type: ignore
    LimitOrder = None  # type: ignore

from sqlalchemy import text
from db import engine  # type: ignore

log = logging.getLogger(__name__)


def connect_ib(
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 1,
) -> Optional["IB"]:
    """
    Establish a connection to Interactive Brokers Trader Workstation.

    Parameters
    ----------
    host : str, optional
        Hostname where TWS/IB Gateway is running. Defaults to ``"127.0.0.1"``.
    port : int, optional
        Port number used by TWS/IB Gateway. Defaults to ``7497`` (paper trading).
    client_id : int, optional
        Client ID for the session. Must be unique among concurrent connections.

    Returns
    -------
    IB or None
        Connected IB instance on success; ``None`` if the connection fails or
        ``ib_insync`` is not available.
    """
    if IB is None:
        log.error(
            "ib_insync is not installed. Please install it to enable IB integration."
        )
        return None
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id)
        if not ib.isConnected():
            raise Exception("IB connection failed.")
        log.info(f"Connected to IB TWS at {host}:{port} with client_id {client_id}")
        return ib
    except Exception as e:
        log.error(f"Failed to connect to IB: {e}")
        return None


def fetch_portfolio_positions(ib: "IB") -> pd.DataFrame:
    """
    Fetch current positions from IB and return as a DataFrame.

    Parameters
    ----------
    ib : IB
        Connected IB instance.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``symbol`` and ``shares``.  Empty if no positions.
    """
    portfolio = ib.portfolio()
    records = []
    for pos in portfolio:
        try:
            symbol = getattr(pos.contract, "symbol", None)
            quantity = getattr(pos, "position", 0)
            # Some positions may be zero or None; skip them.
            if symbol and quantity:
                records.append({"symbol": symbol, "shares": int(quantity)})
        except Exception:
            continue
    return pd.DataFrame(records)


def update_current_positions_from_ib(
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 1,
) -> bool:
    """
    Replace the ``current_positions`` table with live positions from IB.

    This function connects to IB, pulls all open positions, truncates
    the ``current_positions`` table, and inserts the fresh positions.

    Parameters
    ----------
    host : str, optional
        Hostname of TWS/IB Gateway. Defaults to ``"127.0.0.1"``.
    port : int, optional
        Port on which TWS/IB Gateway is listening. Defaults to ``7497``.
    client_id : int, optional
        Client ID for the session. Defaults to ``1``.

    Returns
    -------
    bool
        ``True`` if the update succeeds; ``False`` otherwise.
    """
    ib = connect_ib(host=host, port=port, client_id=client_id)
    if ib is None:
        return False
    try:
        df = fetch_portfolio_positions(ib)
        with engine.begin() as con:
            # Clear existing positions
            con.execute(text("TRUNCATE TABLE current_positions"))
            # Insert new positions
            if not df.empty:
                df.to_sql("current_positions", con, if_exists="append", index=False)
        log.info(f"Updated current_positions with {len(df)} rows from IB.")
        return True
    except Exception as e:
        log.error(f"Failed to update current positions from IB: {e}")
        return False
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def trades_to_ib_orders(
    ib: "IB",
    trades_df: pd.DataFrame,
    use_limit_orders: bool = False,
    price_buffer: float = 0.005,
) -> Dict[int, dict]:
    """
    Convert a trades DataFrame into a mapping of trade IDs to IB order payloads.

    Parameters
    ----------
    ib : IB
        Connected IB instance. Required to construct contracts.
    trades_df : pandas.DataFrame
        DataFrame with columns ``id``, ``symbol``, ``side``, ``quantity``, and ``price``.
    use_limit_orders : bool, optional
        If ``True``, create limit orders with a small price buffer.  Otherwise,
        create market orders.  Defaults to ``False``.
    price_buffer : float, optional
        Percentage buffer (e.g., 0.005 = 0.5%) added or subtracted from the
        reported price when constructing limit orders.  Ignored for market orders.

    Returns
    -------
    dict[int, dict]
        Mapping of trade ID to a dict with keys ``contract`` and ``order``.  The
        payload can be used with `ib.placeOrder`.  If no orders are generated,
        the mapping is empty.
    """
    orders: Dict[int, dict] = {}
    if trades_df is None or trades_df.empty:
        return orders
    for _, row in trades_df.iterrows():
        trade_id = row.get("id")
        symbol = row.get("symbol")
        side = str(row.get("side", "")).upper()
        quantity = int(row.get("quantity", 0))
        price = float(row.get("price", 0.0))
        if not symbol or quantity <= 0:
            continue
        action = "BUY" if side.startswith("B") else "SELL"
        # Define a basic US equity contract on SMART exchange.
        contract = Stock(symbol, "SMART", "USD")
        if use_limit_orders and price > 0:
            delta = price * float(price_buffer)
            limit_price = price + delta if action == "BUY" else price - delta
            order = LimitOrder(action, quantity, round(limit_price, 2))
        else:
            order = MarketOrder(action, quantity)
        # Use trade_id as key; ensure it is integer and not NaN
        try:
            tid = int(trade_id) if trade_id is not None else None
        except Exception:
            tid = None
        if tid is not None:
            orders[tid] = {"contract": contract, "order": order}
    return orders


def submit_ib_orders(
    ib: "IB",
    orders: Dict[int, dict],
) -> Dict[int, str]:
    """
    Submit prepared orders to Interactive Brokers.

    **Warning:** This function will place real or paper orders depending on your
    TWS configuration.  The assistant will not invoke this function on its own.

    Parameters
    ----------
    ib : IB
        Connected IB instance.
    orders : dict[int, dict]
        Mapping from trade ID to order payloads as returned by
        ``trades_to_ib_orders``.

    Returns
    -------
    dict[int, str]
        Mapping of trade ID to a status string (``submitted:<orderId>`` or
        ``error:<message>``).
    """
    results: Dict[int, str] = {}
    for trade_id, payload in orders.items():
        contract = payload.get("contract")
        order = payload.get("order")
        try:
            ib_trade = ib.placeOrder(contract, order)
            # The orderId is available on the trade's order
            results[trade_id] = f"submitted:{ib_trade.order.orderId}"
            log.info(
                f"Submitted IB order for {contract.symbol} with id={ib_trade.order.orderId}"
            )
        except Exception as e:
            results[trade_id] = f"error:{e}"
            log.error(f"Failed to submit IB order for {contract.symbol}: {e}")
    return results
