# Small-Cap Quant System — Institutional Step-Up (Polygon P.I.T.)

This release integrates **Polygon point-in-time fundamentals** (async + retries), **Alpaca SIP** market data with a **Polygon aggregates fallback**, incremental feature engineering, a **multi-model ensemble** with robust imputation & cross-sectional normalization, an **exposure-corrected walk-forward backtest**, and safer **limit-order execution**. It also includes an **EOD pipeline** wired via Render cron.

## Highlights
- **Data**
  - Prices: Alpaca (feed=SIP) batched; fallback to Polygon /aggs (adjusted).
  - Fundamentals: Polygon vX reference financials, **as-of** the filing/period date (leak-free).
  - Async ingestion with bounded concurrency and exponential backoff.
- **Features**
  - Incremental, batched; RSI uses **Wilder’s EWM**; turnover uses adjusted price.
  - Allows sparse fundamentals (ML imputes missing via median).
- **ML**
  - XGB / RF / Ridge + **blend_v1** with configurable weights.
  - Cross-sectional winsorize + z-score per date.
  - Predictions stored **per model** (PK = symbol, ts, model_version).
- **Backtest**
  - Trains full ensemble each tranche; scales daily to target **gross/net**; fills missing returns; applies slippage on open/close.
- **Trading**
  - Generates target positions and trades; safer **limit orders** at arrival price ± 5 bps.
- **Ops**
  - Docker runs **alembic upgrade head** then Streamlit.
  - Render cron jobs include **full EOD pipeline** (`run_pipeline.py`).

## Quickstart
```bash
# Set env (Render: use dashboard)
export DATABASE_URL=postgresql+psycopg://USER:PASS@HOST:PORT/DB
export POLYGON_API_KEY=...

pip install -r requirements.txt
alembic upgrade head
streamlit run app.py
```

## End-to-end (manual)
1. Rebuild universe (sidebar).
2. Backfill market data (7–365 days).
3. Ingest fundamentals (Polygon).
4. Build features (incremental).
5. Train & predict (all models + blend).
6. Generate today's trades; optionally **Sync with Broker**.

## Notes
- Universe filters: `market_cap < 3B`, `ADV_USD_20 > 25k`; configurable via env.
- Backtest and trading respect `GROSS_LEVERAGE`, `NET_EXPOSURE`, and `MAX_POSITION_WEIGHT`.
- See `render.yaml` for scheduled jobs. EOD pipeline runs post-close.
