from __future__ import annotations
import os
import sys
import logging
import traceback
import tempfile
import argparse
import inspect
from datetime import date, datetime, timedelta

log = logging.getLogger(__name__)

def _check_dependencies() -> tuple[bool, str]:
    """Check if all required dependencies are available."""
    missing_deps = []
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    try:
        import sqlalchemy
    except ImportError:
        missing_deps.append("sqlalchemy")
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")

    if missing_deps:
        return False, f"Missing required dependencies: {', '.join(missing_deps)}"
    return True, "All core dependencies available"

def _check_database_connection() -> tuple[bool, str]:
    """Check if database connection is available."""
    try:
        from sqlalchemy import create_engine, text
        from config import DATABASE_URL
        if not DATABASE_URL:
            return False, "DATABASE_URL environment variable not set"
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Database connection verified"
    except Exception as e:
        return False, f"Database check failed: {str(e)}"

def _run_alembic_upgrade() -> tuple[bool, str]:
    """Attempt to run Alembic database migrations."""
    try:
        import subprocess
        import shutil
        alembic_cmd = shutil.which("alembic")
        if not alembic_cmd:
            return False, "Alembic command not found in PATH"
        
        # Use 'heads' to resolve multiple migration branches if they exist.
        result = subprocess.run(
            [alembic_cmd, "upgrade", "heads"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            return True, "Database migrations completed successfully"
        else:
            # Log the full error for debugging
            log.error("Alembic upgrade failed. STDERR:\n%s", result.stderr)
            return False, f"Migration failed. See logs for details."
    except Exception as e:
        return False, f"Migration error: {str(e)}"

def _import_modules() -> tuple[bool, dict, str]:
    """Safely import all pipeline modules."""
    modules = {}
    try:
        from data.ingest import ingest_bars_for_universe
        modules['ingest_bars_for_universe'] = ingest_bars_for_universe
        from data.fundamentals import fetch_fundamentals_for_universe
        modules['fetch_fundamentals_for_universe'] = fetch_fundamentals_for_universe
        from models.features import build_features
        modules['build_features'] = build_features
        from trading.generate_trades import generate_today_trades
        modules['generate_today_trades'] = generate_today_trades
        from trading.broker import sync_trades_to_broker
        modules['sync_trades_to_broker'] = sync_trades_to_broker
        from config import ENABLE_DATA_VALIDATION
        modules['ENABLE_DATA_VALIDATION'] = ENABLE_DATA_VALIDATION
        # Optional ML module
        try:
            from models.ml import train_and_predict_all_models
            modules['train_and_predict_all_models'] = train_and_predict_all_models
        except ImportError as e:
            log.warning(f"Could not import 'models.ml', likely missing optional ML libraries like xgboost. ML steps will be skipped. Error: {e}")
            modules['train_and_predict_all_models'] = None
            
        # Optional enhanced trade generation
        try:
            from enhanced_trade_generation import enhanced_generate_today_trades
            modules['enhanced_generate_today_trades'] = enhanced_generate_today_trades
        except Exception:
            modules['enhanced_generate_today_trades'] = None

        return True, modules, "All modules imported successfully"

    except ImportError as e:
        return False, {}, f"A critical module is missing: {e}"

def _parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()

def _previous_market_day(ref: date) -> date:
    """
    Compute previous market day using market_calendar if available; otherwise fallback to previous weekday.
    """
    try:
        # Try common function names in market_calendar
        from market_calendar import get_previous_market_day as _prev_md  # type: ignore
        return _prev_md(ref)
    except Exception:
        pass
    try:
        from market_calendar import previous_market_day as _prev_md  # type: ignore
        return _prev_md(ref)
    except Exception:
        pass
    # Fallback: previous weekday (does not account for holidays)
    d = ref - timedelta(days=1)
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return d

def _call_with_optional_date(func, target: date | None):
    """
    Call a function, passing a date if it supports one of the common parameter names.
    """
    if target is None:
        return func()
    try:
        sig = inspect.signature(func)
        for pname in ("target_date", "as_of", "run_date", "trade_date", "date"):
            if pname in sig.parameters:
                return func(**{pname: target})
    except Exception:
        # If anything goes wrong with introspection, fall back to calling without args
        pass
    return func()

def main(
    sync_broker: bool = False,
    target_date: date | None = None,
    ignore_market_hours: bool = False,
) -> bool:
    """
    A robust daily pipeline that prioritizes reliability and clear error reporting.
    It relies solely on Alembic for schema management and will fail clearly if the
    database is not ready, preventing execution with a potentially inconsistent state.
    
    Enhanced with:
    - Version guard checking
    - Market calendar validation
    - Enhanced trade generation with abort mechanisms
    - Warning deduplication
    - Target-date override for re-running prior days
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    log.info("🚀 Starting enhanced pipeline (sync_to_broker=%s)", sync_broker)
    if target_date:
        os.environ["PIPELINE_TARGET_DATE"] = target_date.isoformat()
        log.info("📅 Target date override set to %s", target_date.isoformat())
    if ignore_market_hours:
        os.environ["EOD_FORCE_RUN"] = "true"
        log.info("⏱️ Market-hours gate will be ignored for this run")

    # Version Guard Check
    try:
        from version_guard import check_schema_version
        version_ok, version_msg = check_schema_version()
        if not version_ok:
            log.error("❌ Version guard failed: %s", version_msg)
            return False
        log.info("✅ Version guard passed: %s", version_msg)
    except Exception as e:
        log.warning("⚠️ Version guard check failed (proceeding anyway): %s", e)
    
    # Market Calendar Check
    try:
        from market_calendar import should_run_pipeline, log_market_calendar_status

        # Try logging status for the target date if supported
        try:
            sig = inspect.signature(log_market_calendar_status)
            if target_date and ("date" in sig.parameters or "as_of" in sig.parameters or "target_date" in sig.parameters):
                kwargs = {}
                if "date" in sig.parameters:
                    kwargs["date"] = target_date
                elif "as_of" in sig.parameters:
                    kwargs["as_of"] = target_date
                else:
                    kwargs["target_date"] = target_date
                log_market_calendar_status(**kwargs)  # type: ignore
            else:
                log_market_calendar_status()
        except Exception:
            log_market_calendar_status()

        should_run = True
        calendar_reason = "OK to run"

        # If not ignoring market hours, evaluate the gate. For prior-day target we treat it as allowed.
        if not ignore_market_hours:
            if target_date and target_date < date.today():
                should_run = True
                calendar_reason = f"Targeting prior day {target_date.isoformat()} — bypassing 'market not closed yet' gate"
            else:
                # Try passing target_date to should_run_pipeline if supported
                try:
                    sig = inspect.signature(should_run_pipeline)
                    if target_date and ("date" in sig.parameters or "as_of" in sig.parameters or "target_date" in sig.parameters):
                        kwargs = {}
                        if "date" in sig.parameters:
                            kwargs["date"] = target_date
                        elif "as_of" in sig.parameters:
                            kwargs["as_of"] = target_date
                        else:
                            kwargs["target_date"] = target_date
                        should_run, calendar_reason = should_run_pipeline(**kwargs)  # type: ignore
                    else:
                        should_run, calendar_reason = should_run_pipeline()
                except Exception:
                    should_run, calendar_reason = should_run_pipeline()

        if not should_run:
            log.info("📅 Pipeline skipped: %s", calendar_reason)
            return True  # Not an error, just not a trading day
        log.info("✅ Market calendar check passed: %s", calendar_reason)
    except Exception as e:
        log.warning("⚠️ Market calendar check failed (proceeding anyway): %s", e)
    
    # 1. Pre-flight checks
    deps_ok, deps_msg = _check_dependencies()
    if not deps_ok:
        log.error("❌ Pre-flight check failed: %s", deps_msg)
        return False
    log.info("✅ Dependencies check passed.")
    
    db_ok, db_msg = _check_database_connection()
    if not db_ok:
        log.error("❌ Pre-flight check failed: %s", db_msg)
        return False
    log.info("✅ Database connection verified.")

    from sqlalchemy import create_engine, text
    from config import DATABASE_URL
    lock_engine = create_engine(DATABASE_URL)
    lock_conn = lock_engine.connect()
    
    # Use SQLite-compatible advisory lock simulation
    try:
        if "sqlite" in DATABASE_URL.lower():
            # For SQLite, just check if a lock file exists
            import tempfile
            lock_file = os.path.join(tempfile.gettempdir(), "pipeline_lock")
            if os.path.exists(lock_file):
                log.info("Another pipeline is running (SQLite lock file exists)")
                lock_conn.close()
                return True
            else:
                # Create lock file
                with open(lock_file, 'w') as f:
                    f.write(str(os.getpid()))
                got_lock = True
        else:
            # PostgreSQL advisory lock
            got_lock = lock_conn.execute(text("SELECT pg_try_advisory_lock(987654321)")).scalar()
        
        if not got_lock:
            log.info("Another pipeline is running")
            lock_conn.close()
            return True
            
    except Exception as e:
        log.warning(f"Advisory lock failed (proceeding anyway): {e}")
        got_lock = True
    try:
        # 2. Database Migration
        log.info("🔧 Attempting to run database migrations...")
        migration_ok, migration_msg = _run_alembic_upgrade()
        if not migration_ok:
            log.error("❌ Database migration failed: %s", migration_msg)
            log.error("💡 The database schema must be up-to-date before running the pipeline. Please fix the migration issue.")
            return False
        log.info("✅ Database schema is up to date.")

        # 3. Import pipeline components
        import_ok, modules, import_msg = _import_modules()
        if not import_ok:
            log.error("❌ Pipeline failed during module import: %s", import_msg)
            return False
        log.info("✅ All pipeline components imported successfully.")

        if not modules:
            log.warning("⚠️ No pipeline modules available; skipping execution.")
            return True

        try:
            # 4. Execute Pipeline Stages
            log.info("📊 Stage 1: Data ingestion")
            # Ingest a recent window; downstream consumers can use PIPELINE_TARGET_DATE
            modules['ingest_bars_for_universe'](7)
            modules['fetch_fundamentals_for_universe']()
            log.info("✅ Data ingestion completed.")

            log.info("🧮 Stage 2: Feature engineering")
            # Use a small batch size to limit memory usage during feature building
            # (default batch_size=200 helps avoid OOM errors on limited-memory plans)
            # Try to pass a date if supported by build_features
            try:
                _call_with_optional_date(modules['build_features'], target_date)
            except Exception:
                modules['build_features'](batch_size=200)
            else:
                # If build_features accepted a date, still ensure batch size if possible
                try:
                    sig = inspect.signature(modules['build_features'])
                    if "batch_size" in sig.parameters:
                        modules['build_features'](batch_size=200)
                except Exception:
                    pass
            log.info("✅ Feature engineering completed.")

            if modules.get('train_and_predict_all_models'):
                log.info("🤖 Stage 3: Model training and prediction")
                # Try to pass a date if supported
                try:
                    _call_with_optional_date(modules['train_and_predict_all_models'], target_date)
                except Exception:
                    modules['train_and_predict_all_models']()
                log.info("✅ Model training and prediction completed.")
            else:
                log.warning("⚠️ Stage 3: Skipping model training due to missing modules.")

            if modules.get('generate_today_trades'):
                log.info("💰 Stage 4: Enhanced trade generation with validation")
                try:
                    # Prefer enhanced version if available
                    trades = None
                    if modules.get('enhanced_generate_today_trades'):
                        trades = _call_with_optional_date(modules['enhanced_generate_today_trades'], target_date)
                    if trades is None:
                        # Fallback to original implementation
                        trades = _call_with_optional_date(modules['generate_today_trades'], target_date)
                    log.info("✅ Trade generation completed.")
                except Exception as e:
                    log.error("❌ Trade generation failed: %s", e)
                    raise

                if sync_broker and modules.get('sync_trades_to_broker') and trades is not None and not trades.empty:
                    log.info("🔄 Stage 5: Syncing trades to broker")
                    ids = trades["id"].dropna().astype(int).tolist() if "id" in trades.columns else []
                    if ids:
                        modules['sync_trades_to_broker'](ids)
                        log.info("✅ Broker sync completed.")
                    else:
                        log.info("ℹ️ No trades to sync to broker.")

                # After trade generation (and optional broker sync), update the top-N portfolio.
                # This stage computes the latest top-N predictions, applies liquidity and
                # market-cap filters, and logs buy/sell instructions so that a consistent
                # portfolio can be maintained.  It does not place any orders but writes
                # to a CSV log for monitoring.
                log.info("📈 Stage 6: Updating top-N portfolio and logging trades")
                try:
                    from trading.top15_portfolio_tracker import run_daily_update
                    # Use default parameters: n=15, min_adv=1M, max_market_cap=3B
                    run_daily_update()
                    log.info("✅ Top-N portfolio update completed.")
                except Exception as e:
                    log.error("❌ Top-N portfolio update failed: %s", e)
            else:
                log.warning("⚠️ Stage 4/5: Skipping trade generation/sync due to missing modules.")

            log.info("🎉 Pipeline completed successfully")
            return True

        except Exception as e:
            log.exception("💥 Pipeline failed with an unexpected error during execution.")
            return False
        finally:
            # Clean up advisory lock.  Use a fresh connection for the unlock to
            # avoid "SSL SYSCALL error: EOF detected" errors when the original
            # connection has been severed.  Any errors during unlock or close
            # are logged and suppressed so they do not affect the pipeline's exit
            # code.
            try:
                if "sqlite" in DATABASE_URL.lower():
                    lock_file = os.path.join(tempfile.gettempdir(), "pipeline_lock")
                    if os.path.exists(lock_file):
                        os.unlink(lock_file)
                else:
                    # Acquire a new connection solely for unlocking.  This avoids
                    # using a stale connection that may have been dropped by the DB
                    # server during long‑running jobs.
                    with engine.connect() as tmp_conn:
                        tmp_conn.execute(text("SELECT pg_advisory_unlock(987654321)"))
            except Exception as e:
                log.warning(f"Lock cleanup failed (ignored): {e}")
            # Always close the original lock connection if it exists
            try:
                lock_conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the EOD pipeline with optional date overrides.")
    parser.add_argument("--sync-to-broker", action="store_true", help="Also sync generated trades to broker.")
    parser.add_argument("--ignore-market-hours", "--force", dest="ignore_market_hours", action="store_true",
                        help="Bypass market-hours gate and run immediately.")
    parser.add_argument("--yesterday", action="store_true", help="Run as if for the previous market day.")
    parser.add_argument("--target-date", type=_parse_iso_date, help="Run as if for the specified date (YYYY-MM-DD).")

    args = parser.parse_args()

    # Resolve target date
    resolved_target: date | None = None
    if args.target_date:
        resolved_target = args.target_date
    elif args.yesterday:
        resolved_target = _previous_market_day(date.today())

    # Determine broker sync setting
    do_sync_env = os.getenv("SYNC_TO_BROKER", "false").lower() == "true"
    do_sync = do_sync_env or args.sync_to_broker

    try:
        success = main(
            sync_broker=do_sync,
            target_date=resolved_target,
            ignore_market_hours=bool(args.ignore_market_hours),
        )
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        log.warning("⚠️ Pipeline execution interrupted by user.")
        sys.exit(130)
    except KeyboardInterrupt:
        log.warning("⚠️ Pipeline execution interrupted by user.")
        sys.exit(130)
