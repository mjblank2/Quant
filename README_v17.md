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

### Optional adj_close Column Support (Resilient)
The system now provides **fully resilient** handling of the optional `adj_close` column in the `daily_bars` table with automatic detection and transparent fallback:

- **Resilient Operation**: System never fails due to missing `adj_close` column - EOD pipeline, feature builds, risk models, and portfolio optimization all work seamlessly
- **Dynamic Detection**: Automatically detects column presence at runtime using centralized `price_expr()` utility
- **Transparent Fallback**: Uses `COALESCE(adj_close, close)` when column exists, falls back to `close` when missing
- **Consistent DataFrame Schema**: All queries maintain expected column names (e.g., `adj_close` alias) for backward compatibility
- **Single Logging**: One INFO message per process indicates which price mode is active

**Key Benefits:**
- ✅ No more `psycopg.errors.UndefinedColumn` crashes
- ✅ Hot-fix compatible - works immediately without schema changes
- ✅ Forward compatible - automatically uses adjusted prices when column is added
- ✅ Zero code changes required when transitioning between modes

To add `adj_close` column to existing database (optional):
```sql
ALTER TABLE daily_bars ADD COLUMN adj_close numeric;
UPDATE daily_bars SET adj_close = close WHERE adj_close IS NULL;
```

The system will automatically detect and start using adjusted prices on next restart.

### Daily pipeline (unchanged)
```bash
python run_pipeline.py
```

### Addons
```bash
python run_v17_addons.py --snapshot   # write universe snapshot for survivorship-safe training
python run_v17_addons.py --overlays   # propose collars for the latest book
```
