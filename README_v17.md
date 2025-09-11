# Small-Cap Quant System — v17 (PR Bundle)
Differentiated small-cap alpha with **survivorship-safe** training, **PEAD & Russell** event features,
**regime-gated** blending, **tax/lot scaffolding**, optional **QP de-crowding optimizer**, **borrow-aware** hooks,
and **wider options overlays** (collars/put-spreads). Includes a cron **back-compat shim**.

### What changed since v16
- Universe **snapshots** (`universe_history`) + training-time gating
- **PEAD** (`pead_event`, `pead_surprise_*`) and **Russell in/out** events (sparse AltSignals)
- **Regime classifier** (calm/normal/stressed) gates ensemble weights at prediction time
- Optional **cvxpy QP** optimizer with factor-similarity crowding penalty
- **Short borrow** ingestion + helpers
- **Options overlays** (collars) persisted to `option_overlays` + Streamlit `app_v17_panel.py`
- Cron shim: `python -m data_ingestion.run_daily` forwards to `run_pipeline.py`

### Migrations
```bash
alembic upgrade head
```

### Optional adj_close Column Support
The pipeline now supports optional `adj_close` column in the `daily_bars` table with automatic fallback to `close` prices. 

- If `adj_close` column exists: Uses `COALESCE(adj_close, close)` for price queries
- If `adj_close` column is missing: Automatically falls back to `close` column only
- Single INFO log message on startup indicates which mode is active

To add `adj_close` column to existing database:
```sql
ALTER TABLE daily_bars ADD COLUMN adj_close numeric;
UPDATE daily_bars SET adj_close = close WHERE adj_close IS NULL;
```

### Daily pipeline (unchanged)
```bash
python run_pipeline.py
```

### Addons
```bash
python run_v17_addons.py --snapshot   # write universe snapshot for survivorship-safe training
python run_v17_addons.py --overlays   # propose collars for the latest book
```
