# Small-Cap Quant System — v17
**Fast-follows integrated:** survivorship replay (training), PEAD & Russell AltSignals, regime gating, CPCV utilities, upgraded short-borrow, QP optimizer (optional), and expanded options overlays.

### Highlights
- **Universe survivorship replay**: weekly snapshots (`universe_history`) and training-time gating.
- **PEAD & Russell reconstitution**: CSV loaders populate `alt_signals` and `russell_membership`; models consume sparse features automatically.
- **Regime gating**: vol/liquidity classifier modulates ensemble blend weights.
- **CPCV**: utilities for robust out-of-sample validation on time-blocks with embargo.
- **Short-borrow**: PIT borrow table + carry cost helpers.
- **QP optimizer** *(optional)*: convex optimizer with crowding penalty; falls back to greedy if `cvxpy` unavailable.
- **Options overlays**: protective put, put-spread, and collar suggestions stored in `option_overlays`.

### New ENV
```
USE_UNIVERSE_HISTORY=true
REGIME_GATING=true
USE_QP_OPTIMIZER=false
QP_CORR_PENALTY=0.05
UNIVERSE_FILTER_RUSSELL=false
RUSSELL_INDEX=R2000
```
Run new migration:
```
alembic upgrade head
```
Then refresh features/events:
```
python -m models.features
python -m data.events --help  # see loaders
```