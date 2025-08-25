# v18 + MLOps (MLflow + Great Expectations) — Combined Patch
Generated: 2025-08-25T18:37:41.806426Z

## What this is
A single git-style patch that **first** applies the v18 upgrade and **then** wires MLflow + Great Expectations.

## How to apply (recommended: new branch)
```bash
git checkout -b v18-combined
git apply --check v18-combined.patch            # dry-run
git apply --whitespace=fix v18-combined.patch   # apply
pip install -r requirements.txt
alembic upgrade head
python run_pipeline.py
python -c "from validation.wfo import run_wfo; print(run_wfo())"
git add -A && git commit -m "v18 combined: pillars + MLflow/GE"
git push -u origin v18-combined
```
If you use **GitHub Codespaces**: upload this zip to the repo (new branch) → open Codespace → unzip and run the commands above.

## Notes
- Baseline expected: your current v16/v17 tree. If you’ve modified files, resolve any `*.rej` hunks manually, then `git add` and commit.
- If Great Expectations fails hard, set `DATA_VALIDATION_FAIL_HARD=false` in your environment.
- To enable MLflow logging:
  ```bash
  export ENABLE_MLFLOW=true
  export MLFLOW_TRACKING_URI="file:./mlruns"   # or your server URI
  export MLFLOW_EXPERIMENT="smcap-v18"
  ```
