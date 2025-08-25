# Small-Cap Quant System — v16
**Differentiated small-cap alpha, tax-aware portfolio construction, de-crowding, and beta-hedging**

This release focuses on:
- **Differentiated small-cap alpha**: adds microstructure & liquidity premia (Amihud illiquidity), overnight gap, and **rolling beta** features. Enforces sector neutrality (optional) and **market-beta neutrality**.
- **De-crowding**: cross-sectional neutralization vs. common factors (size, momentum, turnover, beta) and sector buckets; correlation & factor-alignment penalties in the optimizer.
- **Tax efficiency**: introduces **tax-lot tracking** (reconstructed from executed trades) with HIFO/FIFO selection, wash-sale awareness, and **sell penalties** for short-term gains; **turnover penalties** baked into the optimizer.
- **Cardinality-controlled book**: Greedy + local-swap optimizer targeting **10–20 long names** with per-sector caps, ADV/price gates, and optional shorts; optional **IWM hedge** to pin portfolio beta near 0.
- **Execution realism**: preserves v15 square-root market-impact (TCA), adds explicit spread & commission knobs in config; stabilizes borrow/no-locate randomness across dates.
- **Quality & robustness**: fixes potential `nanquantile` API change (NumPy ≥1.22), improves Alembic inspector usage, guards log/zero issues in features, and ensures sector-map table existence on use.

> Target profile: concentrated (10–20 longs), small-cap focus, differentiated signals, lower crowding exposure, tax-aware turnover targeting, optional shorts and ETF hedge.
> 
> **Caveat**: >30% annualized is an ambitious target; this code hardens process & alpha hygiene but outcomes depend on data quality, borrow/fees, and execution. Always validate on **PIT** datasets and avoid in-sample bias.

## Quick Start (what changed)
1. Apply the new migration:
   ```bash
   alembic upgrade head
   ```
2. Rebuild features to compute new columns (overnight gap, illiquidity, beta):
   ```bash
   python -m models.features
   ```
3. (Optional) Rebuild tax lots from historical filled trades:
   ```bash
   python -m tax.lots --rebuild
   ```
4. Train/predict and generate trades (now using the tax/crowding-aware optimizer):
   ```bash
   python -m models.ml
   python -m trading.generate_trades
   ```
5. In Streamlit, v16 panel is exposed via `app_v16_panel.py`.

## New ENV / Config knobs
- `TURNOVER_TARGET_ANNUAL=2.5` (x equity / year), `TURNOVER_PENALTY_BPS=25`
- `TAX_LOT_METHOD=hifo|fifo`, `TAX_ST_PENALTY_BPS=150`, `TAX_LT_DAYS=365`, `TAX_WASH_DAYS=30`
- `BETA_HEDGE_SYMBOL=IWM`, `BETA_HEDGE_MAX_WEIGHT=0.20`, `BETA_TARGET=0.0`
- `SPREAD_BPS=7`, `COMMISSION_BPS=0.5`
- `MAX_NAME_CORR=0.85` (optimizer de-duplication)
- `USE_UNIVERSE_HISTORY=true` (recommended already supported in v15)

See `config.py` for defaults.

## What makes it “less crowded”?
- Residualization vs. popular exposures (size/momentum/liquidity/beta + sectors)
- Penalizing names highly correlated with the rest of the book, and factor-aligned picks
- Emphasis on **illiquidity** and **overnight gap** signals that are stronger in micro/small-caps
- Concentration limits per sector while enforcing uniqueness

## Tax approach in brief
- Reconstruct lots from filled trades; maintain open lots table.
- When the optimizer considers *closing or reducing* a position, it computes:
  - **Short-term gain penalty** if the reduction realizes gains within `TAX_LT_DAYS`.
  - **Wash-sale risk penalty** around `TAX_WASH_DAYS` if harvesting losses.
- Optimizer prefers **HIFO** (reduce realized gain) by default; configurable.

## Files added/changed in v16
- **NEW**: `portfolio/optimizer.py`, `tax/lots.py`, `app_v16_panel.py`
- **NEW migration**: `alembic/versions/20250825_06_tax_lots_and_extras.py`
- **UPDATED**: `config.py`, `models/features.py`, `models/ml.py`, `risk/risk_model.py`, `risk/sector.py`, `trading/generate_trades.py`
===== END FILE =====