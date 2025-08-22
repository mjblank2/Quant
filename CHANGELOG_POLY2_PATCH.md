# Poly2 Patch – 2025-08-22 21:18 UTC

## What changed
- Alembic migration typo fixed: `sa Float()` → `sa.Float()`.
- generate_trades: latest price query is now **per-symbol** (join to each symbol's own latest `ts`).
- Backtest P&L: daily rescaling to **gross leverage** and **net exposure** targets before applying returns.
- HTTP client: added **latency/status** logging and exception warnings in `utils_http.get_json()`.

## Safety checks
- Parsed/validated all Python sources successfully (AST).

