from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date, datetime, timezone
from typing import Dict, Iterable
import uuid
from sqlalchemy import text, bindparam
from db import engine, upsert_dataframe, Trade, Position
from config import (
    TOP_N, LONG_TOP_N, SHORT_TOP_N, ALLOW_SHORTS,
    GROSS_LEVERAGE, NET_EXPOSURE, RISK_BUDGET, MAX_POSITION_WEIGHT,
    MIN_PRICE, MIN_ADV_USD, PREFERRED_MODEL
)

def _latest_prices(symbols: Iterable[str]) -> pd.Series:
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        return pd.Series(dtype=float)
    stmt = text("""
        WITH latest AS (
          SELECT symbol, MAX(ts) AS ts
          FROM daily_bars
          WHERE symbol IN :syms
          GROUP BY symbol
        )
        SELECT b.symbol, COALESCE(b.adj_close, b.close) AS px
        FROM daily_bars b
        JOIN latest l ON b.symbol = l.symbol AND b.ts = l.ts
    """).bindparams(bindparam("syms", expanding=True))
    with engine.connect() as con:
        df = pd.read_sql_query(stmt, con, params={"syms": tuple(symbols)})
    return df.set_index("symbol")["px"] if not df.empty else pd.Series(dtype=float)

def _adv20(symbols: Iterable[str]) -> pd.Series:
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        return pd.Series(dtype=float)
    stmt = text("""
        SELECT symbol, adv_usd_20
        FROM universe
        WHERE symbol IN :syms
    """).bindparams(bindparam("syms", expanding=True))
    with engine.connect() as con:
        df = pd.read_sql_query(stmt, con, params={"syms": tuple(symbols)})
    return df.set_index("symbol")["adv_usd_20"] if not df.empty else pd.Series(dtype=float)

def _get_current_weights() -> pd.Series:
    with engine.connect() as con:
        cur_ts = con.execute(text("SELECT MAX(ts) FROM positions")).scalar()
        if not cur_ts:
            return pd.Series(dtype=float)
        df = pd.read_sql_query(
            text("SELECT symbol, weight FROM positions WHERE ts = :ts"),
            con,
            params={"ts": cur_ts}
        )
    return df.set_index("symbol")["weight"] if not df.empty else pd.Series(dtype=float)

def _load_latest_predictions() -> pd.DataFrame:
    with engine.connect() as con:
        preds = pd.read_sql_query(
            text("""
                WITH target AS (
                    SELECT MAX(ts) AS mx FROM predictions WHERE model_version = :mv
                )
                SELECT symbol, ts, y_pred FROM predictions
                WHERE model_version = :mv AND ts = (SELECT mx FROM target)
            """),
            con,
            params={"mv": PREFERRED_MODEL}
        )
        if preds.empty:
            preds = pd.read_sql_query(
                text("SELECT symbol, ts, y_pred FROM predictions WHERE ts = (SELECT MAX(ts) FROM predictions)"),
                con
            )
    return preds

def _compute_side_grosses(gross: float, net: float) -> tuple[float, float]:
    L = max(0.0, (gross + net) / 2.0)
    S = max(0.0, (gross - net) / 2.0)
    return L, S

def generate_today_trades() -> pd.DataFrame:
    preds = _load_latest_predictions()
    if preds.empty:
        raise RuntimeError("No predictions available. Train the model first.")

    preds = preds.sort_values("y_pred", ascending=False)
    long_syms = preds.head(LONG_TOP_N)["symbol"].tolist()
    short_syms: list[str] = []
    if ALLOW_SHORTS:
        short_syms = preds.tail(SHORT_TOP_N)["symbol"].tolist()
        short_syms = [s for s in short_syms if s not in long_syms]

    L_gross, S_gross = _compute_side_grosses(GROSS_LEVERAGE, NET_EXPOSURE if ALLOW_SHORTS else GROSS_LEVERAGE)

    nL, nS = len(long_syms), len(short_syms)
    per_w_L = min(L_gross / nL, MAX_POSITION_WEIGHT) if nL else 0.0
    per_w_S = min(S_gross / nS, MAX_POSITION_WEIGHT) if nS else 0.0

    target_weights: Dict[str, float] = {}
    for s in long_syms:
        target_weights[s] = per_w_L
    for s in short_syms:
        target_weights[s] = -per_w_S

    current = _get_current_weights()
    all_syms = sorted(set(target_weights).union(current.index.tolist()))
    prices = _latest_prices(all_syms)
    adv20 = _adv20(all_syms)

    trade_rows: list[dict] = []
    pos_rows: list[dict] = []
    for sym in all_syms:
        tgt = float(target_weights.get(sym, 0.0))
        cur = float(current.get(sym, 0.0))
        price = float(prices.get(sym, np.nan))
        if not (np.isfinite(price) and price > 0):
            continue

        opening_new = (cur == 0.0) or (np.sign(tgt) != np.sign(cur))
        if opening_new:
            adv_val = float(adv20.get(sym, np.nan))
            if price < MIN_PRICE or (np.isfinite(adv_val) and adv_val < MIN_ADV_USD):
                continue

        shares = int((abs(tgt) * RISK_BUDGET) / max(price, 0.01))
        pos_rows.append({"ts": date.today(), "symbol": sym, "weight": tgt, "price": price, "shares": shares})

        delta_w = tgt - cur
        if abs(delta_w) < 1e-9:
            continue

        notional = RISK_BUDGET * delta_w
        qty = int(abs(notional) / max(price, 0.01))
        if qty <= 0:
            continue

        side = "BUY" if delta_w > 0 else "SELL"
        coid = f"scq-{sym}-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:10]}"
        trade_rows.append({"symbol": sym, "side": side, "quantity": qty, "price": price, "client_order_id": coid})

    if pos_rows:
        pos_df = pd.DataFrame(pos_rows)
        upsert_dataframe(pos_df, Position, ["ts","symbol"])

    if not trade_rows:
        return pd.DataFrame(columns=["symbol","side","quantity","price","status","trade_date"])

    df = pd.DataFrame(trade_rows)
    df["status"] = "generated"
    df["trade_date"] = date.today()
    with engine.begin() as conn:
        conn.execute(Trade.__table__.insert(), df.to_dict(orient="records"))
    return df
