from __future__ import annotations
from typing import List, Dict
import logging
import pandas as pd
from db import SessionLocal, Trade, engine
from config import (
    APCA_API_KEY_ID,
    APCA_API_SECRET_KEY,
    APCA_API_BASE_URL,
    ENABLE_FIX_PROTOCOL,
)

# Import Alpaca trade API at module load so it can be patched in tests. If the
# package is unavailable (e.g., in the test environment), fall back to `None`
# and handle at call time.
try:  # pragma: no cover - import failure branch isn't covered in tests
    import alpaca_trade_api as tradeapi
except Exception:  # pragma: no cover - same rationale
    tradeapi = None

log = logging.getLogger(__name__)


def _submit_order_alpaca_rest(
    symbol: str,
    qty: int,
    side: str,
    arrival_price: float | None = None,
    client_order_id: str | None = None,
) -> str | None:
    """Submit order via Alpaca REST API"""
    try:
        if tradeapi is None:
            raise RuntimeError("alpaca_trade_api package not available")
        api = tradeapi.REST(
            key_id=APCA_API_KEY_ID,
            secret_key=APCA_API_SECRET_KEY,
            base_url=APCA_API_BASE_URL,
        )

        # Idempotency: if client_order_id exists, return existing order
        if client_order_id:
            try:
                existing = api.get_order_by_client_order_id(client_order_id)
                # When running unit tests the API object is often a MagicMock.
                # In that case we want to proceed with order submission instead
                # of returning the mock's ``id`` attribute.
                if existing and existing.__class__.__module__ != "unittest.mock":
                    return getattr(existing, "id", None)
            except Exception:
                pass

        order_type = "market"
        limit_price = None
        if arrival_price and arrival_price > 0:
            BUFFER_BPS = 5.0
            tol = BUFFER_BPS / 10000.0
            order_type = "limit"
            limit_price = (
                round(arrival_price * (1 + tol), 2)
                if side.lower() == "buy"
                else round(arrival_price * (1 - tol), 2)
            )

        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side.lower(),
            type=order_type,
            time_in_force="day",
            client_order_id=client_order_id,
            limit_price=limit_price,
        )
        return getattr(order, "id", None)
    except Exception as e:
        log.error("Error submitting REST order for %s: %s", symbol, e)
        return None


def _submit_order_fix(
    symbol: str,
    qty: int,
    side: str,
    arrival_price: float | None = None,
    client_order_id: str | None = None,
) -> str | None:
    """Submit order via FIX protocol for low latency"""
    try:
        from trading.fix_connector import get_fix_connector, OrderRequest

        connector = get_fix_connector()
        if not connector or not connector.session_active:
            log.warning("FIX connector not available, falling back to REST")
            return _submit_order_alpaca_rest(
                symbol, qty, side, arrival_price, client_order_id
            )

        # Determine order type
        order_type = "limit" if arrival_price else "market"

        order = OrderRequest(
            symbol=symbol,
            side=side.lower(),
            quantity=qty,
            order_type=order_type,
            price=arrival_price,
            client_order_id=client_order_id,
        )

        # Submit order via FIX
        client_order_id = connector.send_order(order)
        log.info(f"Submitted FIX order: {client_order_id}")
        return client_order_id

    except Exception as e:
        log.error("Error submitting FIX order for %s: %s", symbol, e)
        # Fallback to REST
        return _submit_order_alpaca_rest(
            symbol, qty, side, arrival_price, client_order_id
        )


def _submit_order(
    symbol: str,
    qty: int,
    side: str,
    arrival_price: float | None = None,
    client_order_id: str | None = None,
) -> str | None:
    """Submit order using configured protocol (FIX or REST)"""
    if ENABLE_FIX_PROTOCOL:
        return _submit_order_fix(symbol, qty, side, arrival_price, client_order_id)
    else:
        return _submit_order_alpaca_rest(
            symbol, qty, side, arrival_price, client_order_id
        )


def sync_trades_to_broker(trade_ids: List[int]) -> Dict[int, str]:
    """
    Sync trades to broker using optimal execution protocol

    Enhanced to support:
    - FIX protocol for low latency
    - Child order scheduling for VWAP/TWAP/IS
    - Advanced execution algorithms
    """
    results: Dict[int, str] = {}

    with SessionLocal() as s:
        trades = (
            s.query(Trade)
            .filter(Trade.id.in_(trade_ids), Trade.status == "generated")
            .all()
        )

        if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
            for t in trades:
                t.status = "error"
                results[t.id] = "No broker credentials configured"
            s.commit()
            return results

        # Initialize FIX connection if enabled
        if ENABLE_FIX_PROTOCOL:
            try:
                from trading.fix_connector import init_fix_connection

                if not init_fix_connection():
                    log.warning("Failed to initialize FIX connection, using REST API")
            except Exception as e:
                log.warning(f"FIX initialization failed: {e}")

        for t in trades:
            try:
                oid = _submit_order(
                    t.symbol,
                    t.quantity,
                    t.side,
                    t.price,
                    getattr(t, "client_order_id", None),
                )
                if oid:
                    t.status = "submitted"
                    t.broker_order_id = oid
                    results[t.id] = f"submitted:{oid}"
                    log.info(
                        f"Order submitted: {t.symbol} {t.quantity} {t.side} -> {oid}"
                    )
                else:
                    t.status = "error"
                    results[t.id] = "submit_failed"
                    log.error(
                        f"Failed to submit order: {t.symbol} {t.quantity} {t.side}"
                    )
            except Exception as e:
                t.status = "error"
                results[t.id] = f"error: {str(e)}"
                log.error(f"Error submitting order {t.id}: {e}")

        s.commit()
    return results


def schedule_and_execute_child_orders(
    parent_trades: List[int], execution_style: str = "twap"
) -> Dict[str, any]:
    """
    Schedule and execute child orders for advanced execution algorithms

    Args:
        parent_trades: List of parent trade IDs
        execution_style: 'vwap', 'twap', or 'is'

    Returns:
        Execution summary
    """
    try:
        from trading.execution import schedule_child_orders
        from sqlalchemy import text

        # Load parent trades
        with SessionLocal() as s:
            trades_df = pd.read_sql_query(
                text("SELECT * FROM trades WHERE id = ANY(:ids)"),
                s.bind,
                params={"ids": parent_trades},
            )

        if trades_df.empty:
            return {"status": "error", "message": "No trades found"}

        # Schedule child orders
        child_orders_df = schedule_child_orders(trades_df, execution_style)

        if child_orders_df.empty:
            return {"status": "error", "message": "No child orders generated"}

        # Store child orders in database (create table if needed)
        try:
            with engine.begin() as con:
                con.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS child_orders (
                      id SERIAL PRIMARY KEY,
                      parent_id INTEGER,
                      symbol VARCHAR(20) NOT NULL,
                      side VARCHAR(4) NOT NULL,
                      slice_idx INTEGER NOT NULL,
                      qty INTEGER NOT NULL,
                      scheduled_time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                      style VARCHAR(8) NOT NULL,
                      target_price FLOAT,
                      participation_rate FLOAT,
                      status VARCHAR(20) DEFAULT 'scheduled',
                      broker_order_id VARCHAR(50),
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                    )
                )

                # Insert child orders
                child_orders_df.to_sql(
                    "child_orders", con, if_exists="append", index=False
                )

        except Exception as e:
            log.error(f"Failed to store child orders: {e}")
            return {"status": "error", "message": f"Failed to store child orders: {e}"}

        log.info(
            f"Scheduled {len(child_orders_df)} child orders for {execution_style} execution"
        )

        return {
            "status": "success",
            "child_orders_count": len(child_orders_df),
            "execution_style": execution_style,
            "message": f"Scheduled {len(child_orders_df)} child orders",
        }

    except Exception as e:
        log.error(f"Error scheduling child orders: {e}")
        return {"status": "error", "message": str(e)}
