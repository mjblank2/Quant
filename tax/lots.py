from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
import pandas as pd
from sqlalchemy import text
from db import engine
from config import TAX_LOT_METHOD, TAX_ST_PENALTY_BPS, TAX_LT_DAYS, TAX_WASH_DAYS

DDL = """
CREATE TABLE IF NOT EXISTS tax_lots (
  symbol VARCHAR(20) NOT NULL,
  lot_id BIGSERIAL PRIMARY KEY,
  open_date DATE NOT NULL,
  shares INTEGER NOT NULL,
  cost_basis FLOAT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_taxlots_symbol ON tax_lots(symbol);
"""

def _ensure():
    with engine.begin() as con:
        for stmt in DDL.strip().split(';'):
            s = stmt.strip()
            if s:
                con.execute(text(s))

def _load_filled_trades() -> pd.DataFrame:
    with engine.connect() as con:
        df = pd.read_sql_query(text("""
            SELECT trade_date, symbol, side, filled_quantity, avg_fill_price
            FROM trades
            WHERE status IN ('filled','partial_fill') AND filled_quantity IS NOT NULL AND avg_fill_price IS NOT NULL
            ORDER BY trade_date, symbol
        """), con, parse_dates=['trade_date'])
    return df if df is not None else pd.DataFrame(columns=['trade_date','symbol','side','filled_quantity','avg_fill_price'])

def rebuild_tax_lots_from_trades() -> int:
    """Reconstruct open tax lots from recorded filled trades (approx)."""
    _ensure()
    trades = _load_filled_trades()
    if trades.empty:
        return 0
    # reset table
    with engine.begin() as con:
        con.execute(text("DELETE FROM tax_lots"))
    lots = []
    for s, g in trades.groupby('symbol'):
        g = g.sort_values('trade_date')
        # list of (open_date, shares_remaining, cost_basis)
        open_lots = []
        for _, r in g.iterrows():
            qty = int(r['filled_quantity'])
            px  = float(r['avg_fill_price'])
            if r['side'].upper() == 'BUY':
                open_lots.append([r['trade_date'].date(), qty, px])
            else:
                # SELL: match per method
                remain = abs(qty)
                if TAX_LOT_METHOD == 'fifo':
                    open_lots.sort(key=lambda x: x[0])  # oldest first
                else:  # HIFO default
                    open_lots.sort(key=lambda x: x[2], reverse=True)  # highest basis first
                i = 0
                while remain > 0 and i < len(open_lots):
                    if open_lots[i][1] <= remain:
                        remain -= open_lots[i][1]
                        open_lots[i][1] = 0
                        i += 1
                    else:
                        open_lots[i][1] -= remain
                        remain = 0
                open_lots = [lot for lot in open_lots if lot[1] > 0]
        for od, sh, cb in open_lots:
            lots.append({'symbol': s, 'open_date': od, 'shares': int(sh), 'cost_basis': float(cb)})
    if not lots:
        return 0
    df = pd.DataFrame(lots)
    with engine.begin() as con:
        for _, r in df.iterrows():
            con.execute(text("""
                INSERT INTO tax_lots(symbol, open_date, shares, cost_basis)
                VALUES (:s,:d,:q,:cb)
            """), {'s': r['symbol'], 'd': r['open_date'], 'q': int(r['shares']), 'cb': float(r['cost_basis'])})
    return int(len(df))

def _open_lots(symbol: str) -> pd.DataFrame:
    _ensure()
    with engine.connect() as con:
        df = pd.read_sql_query(text("SELECT lot_id, open_date, shares, cost_basis FROM tax_lots WHERE symbol=:s AND shares>0 ORDER BY open_date"), con, params={'s': symbol}, parse_dates=['open_date'])
    return df if df is not None else pd.DataFrame(columns=['lot_id','open_date','shares','cost_basis'])

def tax_sell_penalty_bps(symbol: str, as_of) -> float:
    """
    A rough scalar penalty (bps) for closing a position in `symbol` on `as_of`:
    - penalize short-term gains (held < TAX_LT_DAYS).
    - light penalty for wash-sale risk if harvesting a loss soon after a buy (within TAX_WASH_DAYS).
    Uses *open lots* only; assumes reducing profitable lots first under HIFO, otherwise FIFO.
    """
    lots = _open_lots(symbol)
    if lots.empty:
        return 0.0
    as_of = pd.to_datetime(as_of).date()
    penalty = 0.0
    for _, r in lots.iterrows():
        holding_days = (as_of - r['open_date'].date()).days
        # If lot is in gain territory we don't know unrealized PnL; use heuristic: penalize if < LT threshold
        if holding_days < TAX_LT_DAYS:
            penalty += TAX_ST_PENALTY_BPS * min(1.0, (TAX_LT_DAYS - holding_days) / TAX_LT_DAYS)
        # wash-sale heuristic (weak): if within +-TAX_WASH_DAYS, add small penalty
        if holding_days <= TAX_WASH_DAYS:
            penalty += 10.0
    return penalty
