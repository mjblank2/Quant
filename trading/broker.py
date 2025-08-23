from __future__ import annotations
from typing import List, Dict
from db import SessionLocal, Trade
from config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL
import logging

log = logging.getLogger(__name__)

def _submit_order_alpaca(symbol: str, qty: int, side: str, arrival_price: float | None = None) -> str | None:
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(key_id=APCA_API_KEY_ID, secret_key=APCA_API_SECRET_KEY, base_url=APCA_API_BASE_URL)

        order_type = 'market'
        limit_price = None

        if arrival_price and arrival_price > 0:
            BUFFER_BPS = 5.0
            tolerance = BUFFER_BPS / 10000.0
            order_type = 'limit'
            if side.lower() == 'buy':
                limit_price = round(arrival_price * (1 + tolerance), 2)
            else:
                limit_price = round(arrival_price * (1 - tolerance), 2)
        else:
            log.warning("Submitting MARKET order for %s as no valid price was provided.", symbol)

        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side.lower(),
            type=order_type,
            time_in_force='day',
            limit_price=str(limit_price) if limit_price is not None else None
        )
        return getattr(order, "id", None)
    except Exception as e:
        log.error("Error submitting order for %s (Qty: %d, Side: %s): %s", symbol, qty, side, e)
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
            oid = _submit_order_alpaca(t.symbol, t.quantity, t.side, t.price)
            if oid:
                t.status = "submitted"
                t.broker_order_id = oid
                results[t.id] = f"submitted:{oid}"
            else:
                t.status = "error"
                results[t.id] = "submit_failed"
        s.commit()
    return results
