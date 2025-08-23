# Small-Cap Quant System — Polygon PIT v7

**What’s new vs v6:** 
- Fix **look-ahead leakage** in features: `size_ln` and `turnover_21` are now built from **PIT-safe liquidity proxies** (rolling dollar volume), no current market-cap snapshots.
- **True PIT fundamentals** using `available_at` (SEC acceptance/filing + **T+1 UTC**) and `merge_asof` on availability.
- **Daily rebalance costs** charged when scaling to target gross/net (config: `DAILY_REBALANCE_COST_BPS`, default 1.0 bps).
- **Idempotent orders** to Alpaca via `client_order_id` + numeric limit prices.
- **Nightly pipeline** actually syncs: re-queries DB for `status='generated'` after trade generation before submitting.
- Alembic **base migration**, **driver normalization** to psycopg v3, **UTC discipline**, robust latest price query, **ADV participation cap** on trades, and better HTTP logging.

## Quickstart
```bash
pip install -r requirements.txt
export DATABASE_URL="postgresql+psycopg://USER:PASS@HOST:5432/DB"
export POLYGON_API_KEY="..."
# optional: Alpaca data/broker
export APCA_API_KEY_ID="..."
export APCA_API_SECRET_KEY="..."
alembic upgrade head
streamlit run app.py
```

## Nightly pipeline (Render)
- 21:45 UTC: ingest prices (post-close)
- 22:30 UTC: full pipeline (fundamentals → features → ML → trades → (optional) broker sync)

## Notes
- Predictions PK = (symbol, ts, model_version) with index on (ts, model_version) for ensembles.
- Fundamentals include `as_of` (report period) **and** `available_at` (market knowledge time). Features join on `available_at`.
- Requirements use **psycopg v3** only; `libgomp1` installed in Docker.
