# Small-Cap Quant System — Polygon-First, Import-Safe, and Scalable

**Highlights (this build):**
- ✅ **Polygon fundamentals with true `as_of`** (report/filing period) for **point-in-time** joins — no leakage.
- ✅ **Safe universe rebuild** — upsert new set first then flip only out-of-set names; abort if empty.
- ✅ **Ingestion:** Alpaca **SIP (adjusted)** batch first, **Polygon aggs** fallback (`adjusted=true`). 
- ✅ **Features:** incremental, batched; keep rows unless essential price features missing.
- ✅ **ML:** `SimpleImputer(median)` + XGB/RF/Ridge + blend; predictions keyed by (`symbol`,`ts`,`model_version`).
- ✅ **Backtest:** scale overlapping tranche weights to target **gross leverage** before P&L.
- ✅ **Alembic:** **engine is lazy** (import-safe); migrations won’t fail if `DATABASE_URL` not yet set.
- ✅ **Retries/backoff** around vendor calls; explicit `ts` on trade inserts.

**Run locally**
```bash
pip install -r requirements.txt
alembic upgrade head
streamlit run app.py
```

**Daily automation**
- `run_pipeline.py` orchestrates ingest → fundamentals → features → train/predict → generate trades → (optional) broker sync.

