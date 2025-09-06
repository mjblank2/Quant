# sitecustomize.py
"""
Auto-applied patch for mjblank/quant:
- Wraps db.upsert_dataframe to drop duplicate conflict keys within a single batch
- Prevents Postgres error: ON CONFLICT DO UPDATE command cannot affect row a second time
- Special-cases 'features' by normalizing ts to date for dedupe (since SQL casts ts::DATE)

Env toggles (optional):
  UPSERT_WRAPPER_SILENT=1         -> suppress logs
  UPSERT_KEYS_FEATURES="symbol,ts"-> override keys for features (default same)
"""

import os
import traceback

def _apply_patch():
    try:
        import pandas as pd  # type: ignore
    except Exception:
        # If pandas is unavailable, do nothing (pipeline likely wouldn't run anyway)
        return

    try:
        import db as _db  # your existing module
    except Exception:
        # If user doesn't have db.py or it fails to import, skip patch
        return

    if not hasattr(_db, "upsert_dataframe"):
        return

    _orig_upsert = _db.upsert_dataframe

    def _dedupe_df_for_upsert(df: "pd.DataFrame", table_hint=None, conflict_cols=None):
        # Determine dedupe keys
        keys = None

        # 1) explicit from call
        if isinstance(conflict_cols, (list, tuple)):
            keys = list(conflict_cols)
        elif conflict_cols is not None:
            try:
                # tolerant: convert SQLAlchemy Columns to names if present
                keys = [getattr(c, "name", str(c)) for c in list(conflict_cols)]
            except Exception:
                keys = None

        # 2) env override per table
        if keys is None and isinstance(table_hint, str):
            env_key = f"UPSERT_KEYS_{table_hint.upper()}"
            if env_key in os.environ:
                keys = [s.strip() for s in os.environ[env_key].split(",") if s.strip()]

        # 3) sensible defaults
        if keys is None and isinstance(table_hint, str) and table_hint.lower() == "features":
            keys = ["symbol", "ts"]

        if keys is None and {"symbol", "ts"}.issubset(df.columns):
            keys = ["symbol", "ts"]

        # If no keys resolved or missing in df, bail out quietly
        if not keys or not set(keys).issubset(df.columns):
            return df

        df2 = df.copy()

        # Prefer stable tie-breakers if available
        tie_breakers = [c for c in ("source_ts", "updated_at", "asof", "ingested_at") if c in df2.columns]
        sort_cols = [c for c in keys if c in df2.columns] + tie_breakers
        if sort_cols:
            df2.sort_values(sort_cols, inplace=True)

        # Normalize ts->date for dedupe when present (prevents intraday duplicates collapsing to same DATE)
        subset = list(keys)
        if "ts" in subset and "ts" in df2.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df2["ts"]):
                    df2["__ts_key__"] = df2["ts"].dt.date
                    subset = [("__ts_key__" if c == "ts" else c) for c in subset]
            except Exception:
                # If dtype check fails, keep as-is
                pass

        before = len(df2)
        df2 = df2.drop_duplicates(subset=subset, keep="last")
        dropped = before - len(df2)

        if dropped > 0 and os.environ.get("UPSERT_WRAPPER_SILENT", "0") != "1":
            try:
                # Print a short duplicate summary for CI logs
                dup_df = df.copy()
                if "ts" in dup_df.columns and hasattr(dup_df["ts"], "dt"):
                    try:
                        dup_df["ts_key"] = dup_df["ts"].dt.date
                    except Exception:
                        dup_df["ts_key"] = dup_df["ts"]
                keys_for_group = []
                for k in (keys or []):
                    if k == "ts" and "ts_key" in dup_df.columns:
                        keys_for_group.append("ts_key")
                    elif k in dup_df.columns:
                        keys_for_group.append(k)
                if keys_for_group:
                    dup_summary = (
                        dup_df.groupby(keys_for_group, dropna=False)
                              .size()
                              .reset_index(name="n")
                              .query("n > 1")
                              .head(10)
                    )
                    print(f"[WARN] upsert dedupe: dropped {dropped} duplicate rows on {keys}. Sample:")
                    try:
                        # nice formatting if pandas is present
                        print(dup_summary.to_string(index=False))
                    except Exception:
                        print(dup_summary)
                else:
                    print(f"[WARN] upsert dedupe: dropped {dropped} duplicates (keys {keys} not groupable).")
            except Exception:
                pass

        if "__ts_key__" in df2.columns:
            df2.drop(columns="__ts_key__", inplace=True)

        return df2

    def _wrapper(*args, **kwargs):
        # Find the DataFrame argument (by position or kw)
        df = None
        df_pos = None

        try:
            for i, arg in enumerate(args):
                # Import pandas locally to avoid type issues
                import pandas as pd  # type: ignore
                if isinstance(arg, pd.DataFrame):
                    df = arg
                    df_pos = i
                    break
        except Exception:
            pass

        if df is None:
            df = kwargs.get("df") or kwargs.get("frame") or kwargs.get("dataframe")

        # Best-effort hints
        table_hint = kwargs.get("table") or kwargs.get("table_name") or kwargs.get("name")
        conflict_cols = (
            kwargs.get("conflict_cols")
            or kwargs.get("conflict_columns")
            or kwargs.get("unique_cols")
            or kwargs.get("key_cols")
            or kwargs.get("index_elements")
        )

        if df is not None:
            try:
                df_safe = _dedupe_df_for_upsert(df, table_hint=table_hint, conflict_cols=conflict_cols)
                if df_pos is not None:
                    args = list(args)
                    args[df_pos] = df_safe
                    args = tuple(args)
                else:
                    if "df" in kwargs:
                        kwargs["df"] = df_safe
                    elif "frame" in kwargs:
                        kwargs["frame"] = df_safe
                    elif "dataframe" in kwargs:
                        kwargs["dataframe"] = df_safe
            except Exception:
                # Don't block original call if wrapper has an issue
                if os.environ.get("UPSERT_WRAPPER_SILENT", "0") != "1":
                    traceback.print_exc()

        return _orig_upsert(*args, **kwargs)

    _db.upsert_dataframe = _wrapper
    if os.environ.get("UPSERT_WRAPPER_SILENT", "0") != "1":
        print("[INFO] Patched db.upsert_dataframe with duplicate-safe wrapper (sitecustomize).")

try:
    _apply_patch()
except Exception:
    # Never block pipeline initialization
    traceback.print_exc()
