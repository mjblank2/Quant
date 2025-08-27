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

```bash
# 1) Configure environment (DATABASE_URL, POLYGON_API_KEY, Alpaca keys...)
cp .env.example .env  # edit values

# 2) Build & run locally
pip install -r requirements.txt
alembic upgrade head
streamlit run app.py

# 3) Production (Docker)
docker build -t smallcap-quant:latest .
docker run -e DATABASE_URL=... -e POLYGON_API_KEY=... -p 8501:8501 smallcap-quant:latest
```

## Render Deployment & Cron Jobs

The system is configured for deployment on Render with automated cron jobs for data ingestion and pipeline execution.

### Cron Job Configuration

All cron jobs use direct Python module invocation to avoid shell quoting issues:

```yaml
# ✅ Correct: Direct Python commands
dockerCommand: "python -m data.universe"
dockerCommand: "python -m data.ingest --days 7"  
dockerCommand: "python run_pipeline.py"

# ❌ Avoid: bash -lc wrapping causes quote escaping issues
# dockerCommand: "bash -lc 'python -m data.universe'"
```

**Why direct commands work:**
- The container entrypoint (`scripts/entrypoint.sh`) automatically runs `alembic upgrade head` before executing any dockerCommand
- No need for bash -lc wrapping or complex quote escaping
- Simpler, more reliable execution in Render's container environment

### Local Testing

Test cron commands locally using Docker:

```bash
# Build container
docker build -t smallcap-quant .

# Test cron commands (entrypoint runs Alembic first)
docker run --rm -e DATABASE_URL=... -e POLYGON_API_KEY=... smallcap-quant python -m data.universe
docker run --rm -e DATABASE_URL=... -e POLYGON_API_KEY=... smallcap-quant python -m data.ingest --days 7
docker run --rm -e DATABASE_URL=... -e POLYGON_API_KEY=... smallcap-quant python run_pipeline.py
```

## Notes

- Fundamentals are fetched from Polygon's `vX/reference/financials`; we store **as_of** using the filing/period date and join with `merge_asof(direction='backward')` to avoid look-ahead.
- Prices use Alpaca batch bars (adjusted, **SIP** feed if entitled), then Polygon aggs, then Tiingo.
- Universe build is **atomic**: upsert the new set first, then flip any prior names to `included = FALSE` in the same transaction.
- `run_pipeline.py` executes the full EOD flow for cron.

> ⚠️ Trading is risky; paper trade first. Ensure entitlements for SIP data and API quotas are appropriate for your universe.
