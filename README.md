
# Consolidated Code Drop

This folder contains the consolidated modules discussed, split into packages so you can upload directly to GitHub.

## What you get

- `data_ingestion/dashboard.py` — Streamlit dashboard with robust DB handling
- `data/timescale.py` — Safe TimescaleDB setup using SAVEPOINTs
- `data/ingest.py` — Ingestion scaffolding with improved logging
- `data/validation.py` — Parameterized anomaly detection (SQL-injection safe)
- `data/universe.py` — Universe scaffolding + logging
- `data/universe_history.py` — Survivorship gating stub
- `features/build_features.py` — **Duplicate-safe** feature builder (fixes ON CONFLICT error)
- `features/registry.py` — Feature registry
- `features/store.py` — Feature store
- `features/transformations.py` — Reusable transformers
- `models/transformers.py` — Stateless cross-sectional normalizer
- `models/regime.py` — Regime classification + blend gating
- `models/train_predict.py` — Live training, predictions, backtest
- `portfolio/mvo.py` — Convex optimizer (MVO) with liquidity/beta constraints
- `portfolio/build.py` — Heuristic portfolio builder with optional QP sizing
- `portfolio/qp_optimizer.py` — Small QP helper
- `ml/cpcv.py` — **True CPCV splitter** (function + sklearn splitter)
- `worker.py` — Simple CLI worker wrapper
- `tasks/celery_app.py` — Celery placeholder

## Notes
- These modules rely on your existing `db.py` and `config.py`.
- To fix the `ON CONFLICT DO UPDATE ... cannot affect row a second time` error, the feature builder now **de-duplicates** rows by `(symbol, ts)` before upserting.
- The CPCV splitter in `ml/cpcv.py` returns **row-level** indices compatible with scikit-learn; it purges training rows within an embargo window around each test block.

