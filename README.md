# Small-Cap Quant System — Polygon PTI + Ensemble (v8)

**Highlights**

- **Polygon-first**: prices & *point-in-time* fundamentals (no leakage) with retries/backoff.
- **Incremental features**: momentum/vol/RSI (Wilder), turnover (using adjusted prices), size, + fundamentals.
- **Multi-model**: XGB / RF / Ridge pipelines with `SimpleImputer` + `StandardScaler`, plus blended `blend_v1`.
- **Backtest**: walk-forward, tranche rebalances, *daily exposure scaling* to target gross/net; robust to missing prints.
- **Trade gen**: uses preferred model, ADV/price gates, writes positions & trades; broker sync via **limit** orders.
- **DB & Migrations**: SQLAlchemy 2.0, Alembic. `predictions` PK is now `(symbol, ts, model_version)`.
- **Ops**: Dockerized; Render cron runs ingestion, weekly universe refresh, and full EOD pipeline.

## Quickstart

### Local Development
```bash
# 1) Configure environment (DATABASE_URL, POLYGON_API_KEY, Alpaca keys...)
cp .env.example .env  # edit values

# 2) Build & run locally
pip install -r requirements.txt
alembic upgrade head
streamlit run app.py
```

### Deploy to Render (Recommended)
```bash
# 1) Use our comprehensive deployment guide
# See: DEPLOY_TO_RENDER.md for complete instructions

# 2) Quick deployment checklist:
python scripts/verify_render_setup.py  # Verify setup
# Then deploy using render.yaml blueprint in Render dashboard
```

### Docker Deployment
```bash
# 3) Production (Docker)
docker build -t smallcap-quant:latest .
docker run -e DATABASE_URL=... -e POLYGON_API_KEY=... -p 8501:8501 smallcap-quant:latest
```

## Notes

- Fundamentals are fetched from Polygon's `vX/reference/financials`; we store **as_of** using the filing/period date and join with `merge_asof(direction='backward')` to avoid look-ahead.
- Prices use Alpaca batch bars (adjusted, **SIP** feed if entitled), then Polygon aggs, then Tiingo.
- Universe build is **atomic**: upsert the new set first, then flip any prior names to `included = FALSE` in the same transaction.
- `run_pipeline.py` executes the full EOD flow for cron.

> ⚠️ Trading is risky; paper trade first. Ensure entitlements for SIP data and API quotas are appropriate for your universe.
