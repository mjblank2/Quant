# Small-Cap Quant System (Live Dashboard)

A database-backed small-cap quant stack with a Streamlit control panel and optional Alpaca order submission.

> ⚠️ Trading is risky. Use at your own risk. Paper trade first. This project is for educational purposes.

## What's new in this release

- **Scalable features**: `build_features()` is **incremental and batched**. It computes only for symbols/dates that are new, with a warm-up window for rolling features.
- **Adjusted prices**: Ingestion stores `adj_close` (when available) and features/returns use **adjusted** prices; volume/turnover still use raw prices.
- **Safer universe refresh**: We now mark all rows `included=FALSE` and then **upsert** the fresh list, avoiding a momentary empty universe.
- **Trade targets**: `generate_trades()` writes today's **target positions** via the same `upsert_dataframe` helper for consistency.
- **Render cron**: Weekday ingestion at **21:30 UTC** (post close). Universe refresh Mondays **14:00 UTC**.

## Quickstart

1. Set env vars (`.env`): `DATABASE_URL`, `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, optional `TIINGO_API_KEY`.
2. Install: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

## Notes

- Alpaca Market Data: `ALPACA_DATA_FEED=iex` (or `sip` with entitlements).
- All SQL queries are parameterized; key tables have useful indexes.
- Backtest remains a simple monthly re-train placeholder.
