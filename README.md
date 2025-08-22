# Small-Cap Quant System (Live Dashboard)

A database-backed small-cap quant stack with a Streamlit control panel and optional Alpaca order submission.

## What's in this release

- **Scalable features**: Incremental, batched feature builds with a warm-up window (no full-table loads).
- **Adjusted prices**: Ingestion stores `adj_close` and modeling uses `COALESCE(adj_close, close)` for returns/momentum.
- **Safer universe refresh**: Mark old rows `included=false` and upsert the fresh list (never an empty universe).
- **Trade targets**: `generate_trades()` writes today's **target positions** via upsert; trades use direct insert (autoincrement id).
- **Cron CLIs**: `python -m data.ingest --days 7` and `python -m data.universe` now work.
- **Render-ready**: Dockerfile binds to `$PORT` correctly; `libgomp1` installed for XGBoost; Streamlit headless envs set.
- **Post-close ingest**: Weekdays **21:30 UTC**; Universe refresh Mondays **14:00 UTC**.

## Quickstart

1. Set env vars (`.env`): `DATABASE_URL`, `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, optional `TIINGO_API_KEY`.
2. Install: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

## Notes

- Alpaca Market Data: `ALPACA_DATA_FEED=sip` (or `sip` with entitlements).
- All SQL queries are parameterized; key tables have useful indexes.
- Backtest is a simple monthly re-train placeholder; extend as needed.
