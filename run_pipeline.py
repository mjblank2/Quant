from __future__ import annotations
import os
import sys
import logging
import traceback
import tempfile
from datetime import date

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
            
        return True, modules, "All modules imported successfully"

    except ImportError as e:
        return False, {}, f"A critical module is missing: {e}"


def main(sync_broker: bool = False) -> bool:
    """
    A robust daily pipeline that prioritizes reliability and clear error reporting.
    It relies solely on Alembic for schema management and will fail clearly if the
    database is not ready, preventing execution with a potentially inconsistent state.
    
    Enhanced with:
    - Version guard checking
    - Market calendar validation
    - Enhanced trade generation with abort mechanisms
    - Warning deduplication
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    log.info("🚀 Starting enhanced pipeline (sync_to_broker=%s)", sync_broker)
    
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
        log_market_calendar_status()
        
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
            modules['ingest_bars_for_universe'](7)
            modules['fetch_fundamentals_for_universe']()
            log.info("✅ Data ingestion completed.")

            log.info("🧮 Stage 2: Feature engineering")
            # Use a small batch size to limit memory usage during feature building
            # (default batch_size=200 helps avoid OOM errors on limited-memory plans)
            modules['build_features'](batch_size=200)
            log.info("✅ Feature engineering completed.")

            if modules.get('train_and_predict_all_models'):
                log.info("🤖 Stage 3: Model training and prediction")
                modules['train_and_predict_all_models']()
                log.info("✅ Model training and prediction completed.")
            else:
                log.warning("⚠️ Stage 3: Skipping model training due to missing modules.")

            if modules.get('generate_today_trades'):
                log.info("💰 Stage 4: Enhanced trade generation with validation")
                try:
                    # Use enhanced trade generation with abort mechanisms
                    from enhanced_trade_generation import enhanced_generate_today_trades
                    trades = enhanced_generate_today_trades()
                    log.info("✅ Enhanced trade generation completed.")
                except Exception as e:
                    log.error("❌ Enhanced trade generation failed, falling back to basic: %s", e)
                    # Fallback to original implementation
                    trades = modules['generate_today_trades']()
                    log.info("✅ Fallback trade generation completed.")

                if sync_broker and modules.get('sync_trades_to_broker') and trades is not None and not trades.empty:
                    log.info("🔄 Stage 5: Syncing trades to broker")
                    ids = trades["id"].dropna().astype(int).tolist() if "id" in trades.columns else []
                    if ids:
                        modules['sync_trades_to_broker'](ids)
                        log.info("✅ Broker sync completed.")
                    else:
                        log.info("ℹ️ No trades to sync to broker.")
            else:
                log.warning("⚠️ Stage 4/5: Skipping trade generation/sync due to missing modules.")

            log.info("🎉 Pipeline completed successfully")
            return True

        except Exception as e:
            log.exception("💥 Pipeline failed with an unexpected error during execution.")
            return False
    finally:
        # Clean up lock
        try:
            if "sqlite" in DATABASE_URL.lower():
                lock_file = os.path.join(tempfile.gettempdir(), "pipeline_lock")
                if os.path.exists(lock_file):
                    os.unlink(lock_file)
            else:
                lock_conn.execute(text("SELECT pg_advisory_unlock(987654321)"))
        except Exception as e:
            log.warning(f"Lock cleanup failed: {e}")
        finally:
            try:
                lock_conn.close()
            except:
                pass


if __name__ == "__main__":
    do_sync = os.getenv("SYNC_TO_BROKER", "false").lower() == "true"
    
    try:
        success = main(sync_broker=do_sync)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        log.warning("⚠️ Pipeline execution interrupted by user.")
        sys.exit(130)
