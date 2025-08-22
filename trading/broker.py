from __future__ import annotations
from typing import List, Dict
from db import SessionLocal, Trade
from config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL

def _submit_order_alpaca(symbol: str, qty: int, side: str) -> str | None:
    try:
        import alpaca_trade_api as tradeapi
        from alpaca_trade_api.rest import APIError
        api = tradeapi.REST(key_id=APCA_API_KEY_ID, secret_key=APCA_API_SECRET_KEY, base_url=APCA_API_BASE_URL)
        order = api.submit_order(symbol=symbol, qty=qty, side=side.lower(), type='market', time_in_force='day')
        return getattr(order, "id", None)
    except APIError:
        return None
    except Exception:
        return None

def sync_trades_to_broker(trade_ids: List[int]) -> Dict[int, str]:
    results: Dict[int, str] = {}
    with SessionLocal() as s:
        trades = s.query(Trade).filter(Trade.id.in_(trade_ids)).all()

        if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
            for t in trades:
                t.status = "error"
                results[t.id] = "No broker credentials configured"
            s.commit()
            return results

        for t in trades:
            oid = _submit_order_alpaca(t.symbol, t.quantity, t.side)
            if oid:
                t.status = "submitted"
                t.broker_order_id = oid
                results[t.id] = f"submitted:{oid}"
            else:
                t.status = "error"
                results[t.id] = "submit_failed"
        s.commit()
    return results
