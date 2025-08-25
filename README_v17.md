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

### Daily pipeline (unchanged)
```bash
python run_pipeline.py
```

### Addons
```bash
python run_v17_addons.py --snapshot   # write universe snapshot for survivorship-safe training
python run_v17_addons.py --overlays   # propose collars for the latest book
```
