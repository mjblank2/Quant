from __future__ import annotations
import os
import logging
from data.ingest import ingest_bars_for_universe
from data.fundamentals import fetch_fundamentals_for_universe
from data.institutional_ingest import setup_infrastructure, run_infrastructure_health_check
from data.validation import run_validation_pipeline
from models.features import build_features
from models.ml import train_and_predict_all_models
from trading.generate_trades import generate_today_trades
from trading.broker import sync_trades_to_broker
from config import ENABLE_DATA_VALIDATION

log = logging.getLogger(__name__)

def main(sync_broker: bool = False):
    """Main pipeline with institutional-grade data infrastructure."""
    log.info("Starting institutional pipeline")
    
    # Phase 1: Infrastructure setup and health check
    log.info("Phase 1: Setting up data infrastructure")
    setup_infrastructure()
    
    health_status = run_infrastructure_health_check()
    log.info(f"Infrastructure health: {health_status['overall_status']}")
    
    if health_status['overall_status'] == 'CRITICAL':
        log.error("Critical infrastructure issues detected, aborting pipeline")
        return False
    
    # Phase 2: Data ingestion with validation
    log.info("Phase 2: Data ingestion")
    ingest_bars_for_universe(7)
    fetch_fundamentals_for_universe()
    
    # Phase 3: Data validation
    if ENABLE_DATA_VALIDATION:
        log.info("Phase 3: Data validation")
        validation_result = run_validation_pipeline()
        if validation_result.errors:
            log.error("Data validation failed, continuing with warnings")
            for error in validation_result.errors:
                log.error(f"Validation error: {error}")
    
    # Phase 4: Feature engineering and modeling
    log.info("Phase 4: Feature engineering and modeling")
    build_features()
    outs = train_and_predict_all_models()
    
    # Phase 5: Trade generation and execution
    log.info("Phase 5: Trade generation")
    trades = generate_today_trades()
    
    if sync_broker:
        try:
            ids = trades["id"].dropna().astype(int).tolist() if "id" in trades.columns else []
        except Exception:
            ids = []
        if ids:
            sync_trades_to_broker(ids)
    
    log.info("Pipeline completed successfully")
    return True


if __name__ == "__main__":
    do_sync = os.getenv("SYNC_TO_BROKER", "false").lower() == "true"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Starting pipeline (sync_to_broker=%s)", do_sync)
    main(sync_broker=do_sync)
