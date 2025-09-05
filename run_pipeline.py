from __future__ import annotations
import os
import sys
import logging
import traceback

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
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    log.info("🚀 Starting pipeline (sync_to_broker=%s)", sync_broker)
    
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
            log.info("💰 Stage 4: Trade generation")
            trades = modules['generate_today_trades']()
            log.info("✅ Trade generation completed.")

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
