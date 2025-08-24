from __future__ import annotations
from typing import List, Dict
import logging
from db import SessionLocal, Trade
from config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL

log = logging.getLogger(__name__)

def _submit_order_alpaca(symbol: str, qty: int, side: str, arrival_price: float | None = None, client_order_id: str | None = None) -> str | None:
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(key_id=APCA_API_KEY_ID, secret_key=APCA_API_SECRET_KEY, base_url=APCA_API_BASE_URL)

        # Idempotency: if client_order_id exists, return existing order
        if client_order_id:
            try:
                existing = api.get_order_by_client_order_id(client_order_id)
                return getattr(existing, 'id', None)
            except Exception:
                pass

        order_type = 'market'
        limit_price = None
        if arrival_price and arrival_price > 0:
            BUFFER_BPS = 5.0
            tol = BUFFER_BPS / 10000.0
            order_type = 'limit'
            limit_price = round(arrival_price * (1 + tol), 2) if side.lower() == 'buy' else round(arrival_price * (1 - tol), 2)

        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side.lower(),
            type=order_type,
            time_in_force='day',
            client_order_id=client_order_id,
            limit_price=limit_price
        )
        return getattr(order, "id", None)
    except Exception as e:
        log.error("Error submitting order for %s: %s", symbol, e)
        return None

def sync_trades_to_broker(trade_ids: List[int]) -> Dict[int, str]:
    results: Dict[int, str] = {}
    with SessionLocal() as s:
        trades = s.query(Trade).filter(Trade.id.in_(trade_ids), Trade.status == 'generated').all()

        if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
            for t in trades:
                t.status = "error"
                results[t.id] = "No broker credentials configured"
            s.commit()
            return results

        for t in trades:
            oid = _submit_order_alpaca(t.symbol, t.quantity, t.side, t.price, getattr(t, 'client_order_id', None))
            if oid:
                t.status = "submitted"
                t.broker_order_id = oid
                results[t.id] = f"submitted:{oid}"
            else:
                t.status = "error"
                results[t.id] = "submit_failed"
        s.commit()
    return results
