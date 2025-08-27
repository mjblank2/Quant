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
    """Check if database connection is available and basic tables exist."""
    try:
        from sqlalchemy import create_engine, text
        from config import DATABASE_URL
        
        if not DATABASE_URL:
            return False, "DATABASE_URL environment variable not set"
        
        engine = create_engine(DATABASE_URL)
        
        # Test basic connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Check if basic tables exist
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='universe'"))
            if result.fetchone() is None:
                return False, "Database tables not found - migrations may need to be run"
        
        return True, "Database connection and basic tables verified"
        
    except Exception as e:
        return False, f"Database check failed: {str(e)}"

def _run_alembic_upgrade() -> tuple[bool, str]:
    """Attempt to run Alembic database migrations."""
    try:
        import subprocess
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            return True, "Database migrations completed successfully"
        else:
            return False, f"Migration failed: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "Migration timeout (>60s)"
    except FileNotFoundError:
        return False, "Alembic not found"
    except Exception as e:
        return False, f"Migration error: {str(e)}"

def _create_basic_database_schema() -> tuple[bool, str]:
    """Create basic database schema if migrations fail."""
    try:
        from sqlalchemy import create_engine, text
        from config import DATABASE_URL
        
        engine = create_engine(DATABASE_URL)
        
        # Create basic tables needed for pipeline
        with engine.connect() as conn:
            # Create universe table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS universe (
                    symbol TEXT PRIMARY KEY,
                    included BOOLEAN DEFAULT TRUE,
                    first_date DATE,
                    last_date DATE,
                    market_cap REAL,
                    adv_usd REAL
                )
            """))
            
            # Create daily_bars table  
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS daily_bars (
                    symbol TEXT,
                    ts DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    vwap REAL,
                    trade_count INTEGER,
                    PRIMARY KEY (symbol, ts)
                )
            """))
            
            # Create features table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS features (
                    symbol TEXT,
                    ts DATE,
                    ret_1d REAL,
                    ret_5d REAL,
                    ret_21d REAL,
                    vol_21 REAL,
                    size_ln REAL,
                    PRIMARY KEY (symbol, ts)
                )
            """))
            
            # Create trades table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    quantity INTEGER,
                    side TEXT,
                    price REAL,
                    status TEXT DEFAULT 'generated',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Insert a default universe entry for testing
            conn.execute(text("""
                INSERT OR IGNORE INTO universe (symbol, included, market_cap, adv_usd) 
                VALUES ('AAPL', TRUE, 1000000000, 50000000)
            """))
            
            conn.commit()
        
        return True, "Basic database schema created successfully"
        
    except Exception as e:
        return False, f"Failed to create basic schema: {str(e)}"

def _import_modules() -> tuple[bool, dict, str]:
    """Safely import all pipeline modules."""
    modules = {}
    try:
        from data.ingest import ingest_bars_for_universe
        modules['ingest_bars_for_universe'] = ingest_bars_for_universe
        log.info("✅ data.ingest module imported successfully")
    except Exception as e:
        return False, {}, f"Failed to import data.ingest: {str(e)}"
    
    try:
        from data.fundamentals import fetch_fundamentals_for_universe
        modules['fetch_fundamentals_for_universe'] = fetch_fundamentals_for_universe
        log.info("✅ data.fundamentals module imported successfully")
    except Exception as e:
        return False, {}, f"Failed to import data.fundamentals: {str(e)}"
    
    try:
        from models.features import build_features
        modules['build_features'] = build_features
        log.info("✅ models.features module imported successfully")
    except Exception as e:
        return False, {}, f"Failed to import models.features: {str(e)}"
    
    try:
        from models.ml import train_and_predict_all_models
        modules['train_and_predict_all_models'] = train_and_predict_all_models
        log.info("✅ models.ml module imported successfully")
    except Exception as e:
        log.warning(f"⚠️ models.ml import failed (may be due to missing xgboost): {str(e)}")
        modules['train_and_predict_all_models'] = None
    
    try:
        from trading.generate_trades import generate_today_trades
        modules['generate_today_trades'] = generate_today_trades
        log.info("✅ trading.generate_trades module imported successfully")
    except Exception as e:
        log.warning(f"⚠️ trading.generate_trades import failed: {str(e)}")
        modules['generate_today_trades'] = None
    
    try:
        from trading.broker import sync_trades_to_broker
        modules['sync_trades_to_broker'] = sync_trades_to_broker
        log.info("✅ trading.broker module imported successfully")
    except Exception as e:
        log.warning(f"⚠️ trading.broker import failed: {str(e)}")
        modules['sync_trades_to_broker'] = None
    
    try:
        from config import ENABLE_DATA_VALIDATION
        modules['ENABLE_DATA_VALIDATION'] = ENABLE_DATA_VALIDATION
        log.info("✅ config module imported successfully")
    except Exception as e:
        return False, {}, f"Failed to import config: {str(e)}"
    
    return True, modules, "All modules imported successfully"


def main(sync_broker: bool = False) -> bool:
    """
    Minimal, reliable daily pipeline:
      - Ingest prices and fundamentals
      - Build features
      - Train/predict all models
      - Generate trades
      - Optionally sync to broker
    If ENABLE_DATA_VALIDATION is True, run light GE checks if the optional
    data_validation package is present.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    # Check required environment variables
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        log.error("❌ DATABASE_URL environment variable is required")
        return False
    
    log.info("🚀 Starting pipeline (sync_to_broker=%s)", sync_broker)
    log.info("🔧 DATABASE_URL: %s", database_url[:20] + "..." if len(database_url) > 20 else database_url)
    
    # Check dependencies first
    deps_ok, deps_msg = _check_dependencies()
    if not deps_ok:
        log.error("❌ Dependency check failed: %s", deps_msg)
        return False
    
    log.info("✅ Dependencies check passed: %s", deps_msg)
    
    # Check database connection and tables
    db_ok, db_msg = _check_database_connection()
    if not db_ok:
        log.warning("⚠️ Database check failed: %s", db_msg)
        log.info("🔧 Attempting to run database migrations...")
        
        migration_ok, migration_msg = _run_alembic_upgrade()
        if migration_ok:
            log.info("✅ Database migrations completed: %s", migration_msg)
            # Re-check database after migration
            db_ok, db_msg = _check_database_connection()
            if not db_ok:
                log.error("❌ Database still not ready after migration: %s", db_msg)
                return False
        else:
            log.error("❌ Database migration failed: %s", migration_msg)
            log.info("🔧 Attempting to create basic database schema...")
            
            schema_ok, schema_msg = _create_basic_database_schema()
            if schema_ok:
                log.info("✅ Basic database schema created: %s", schema_msg)
                # Re-check database after schema creation
                db_ok, db_msg = _check_database_connection()
                if not db_ok:
                    log.error("❌ Database still not ready after schema creation: %s", db_msg)
                    return False
            else:
                log.error("❌ Failed to create basic schema: %s", schema_msg)
                log.error("💡 Manual database setup is required")
                return False
    
    log.info("✅ Database check passed: %s", db_msg)
    
    # Import modules safely
    import_ok, modules, import_msg = _import_modules()
    if not import_ok:
        log.error("❌ Module import failed: %s", import_msg)
        return False
    
    log.info("✅ Module imports successful")
    
    try:
        # Phase 1: Ingestion
        log.info("📊 Phase 1: Data ingestion")
        try:
            modules['ingest_bars_for_universe'](7)
            log.info("✅ Bar ingestion completed")
        except Exception as e:
            log.error("❌ Bar ingestion failed: %s", str(e))
            return False
        
        try:
            modules['fetch_fundamentals_for_universe']()
            log.info("✅ Fundamentals ingestion completed")
        except Exception as e:
            log.error("❌ Fundamentals ingestion failed: %s", str(e))
            return False

        # Optional: Data validation (soft-fail unless explicitly configured to fail hard)
        if modules['ENABLE_DATA_VALIDATION']:
            log.info("🔍 Running data validation checks")
            try:
                from data_validation.gx_checks import validate_recent_daily_bars
                validate_recent_daily_bars(days=5, fail_hard=False)
                log.info("✅ Data validation passed")
            except Exception as e:
                log.warning("⚠️ Data validation skipped or failed softly: %s", e)

        # Phase 2: Features and modeling
        log.info("🧮 Phase 2: Feature engineering and modeling")
        try:
            modules['build_features']()
            log.info("✅ Feature engineering completed")
        except Exception as e:
            log.error("❌ Feature engineering failed: %s", str(e))
            return False
        
        # Model training (optional if xgboost not available)
        if modules['train_and_predict_all_models'] is not None:
            try:
                _ = modules['train_and_predict_all_models']()
                log.info("✅ Model training and prediction completed")
            except Exception as e:
                log.error("❌ Model training failed: %s", str(e))
                return False
        else:
            log.warning("⚠️ Skipping model training (dependencies not available)")

        # Phase 3: Trade generation (+ optional sync)
        log.info("💰 Phase 3: Trade generation")
        trades = None
        if modules['generate_today_trades'] is not None:
            try:
                trades = modules['generate_today_trades']()
                log.info("✅ Trade generation completed")
            except Exception as e:
                log.error("❌ Trade generation failed: %s", str(e))
                return False
        else:
            log.warning("⚠️ Skipping trade generation (dependencies not available)")

        if sync_broker and modules['sync_trades_to_broker'] is not None and trades is not None:
            log.info("🔄 Syncing trades to broker")
            try:
                ids = trades["id"].dropna().astype(int).tolist() if "id" in trades.columns else []
            except Exception:
                ids = []
            if ids:
                try:
                    modules['sync_trades_to_broker'](ids)
                    log.info("✅ Broker sync completed")
                except Exception as e:
                    log.error("❌ Broker sync failed: %s", str(e))
                    return False
            else:
                log.info("ℹ️ No trades to sync to broker")

        log.info("🎉 Pipeline completed successfully")
        return True
        
    except Exception as e:
        log.exception("💥 Pipeline failed with unexpected error: %s", e)
        log.error("📋 Full traceback:")
        log.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    do_sync = os.getenv("SYNC_TO_BROKER", "false").lower() == "true"
    
    # Initialize logging first
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    log.info("🎯 Starting pipeline execution (sync_to_broker=%s)", do_sync)
    
    try:
        success = main(sync_broker=do_sync)
        if success:
            log.info("✅ Pipeline execution completed successfully")
            sys.exit(0)
        else:
            log.error("❌ Pipeline execution failed")
            sys.exit(1)
    except KeyboardInterrupt:
        log.warning("⚠️ Pipeline execution interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        log.exception("💥 Unexpected error during pipeline execution: %s", e)
        sys.exit(1)
