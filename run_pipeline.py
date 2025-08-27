from __future__ import annotations
import os
import logging

from data.ingest import ingest_bars_for_universe
from data.fundamentals import fetch_fundamentals_for_universe
from models.features import build_features
from models.ml import train_and_predict_all_models
from trading.generate_trades import generate_today_trades
from trading.broker import sync_trades_to_broker
from config import ENABLE_DATA_VALIDATION

log = logging.getLogger(__name__)


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
    try:
        # Phase 1: Ingestion
        log.info("Phase 1: Data ingestion")
        ingest_bars_for_universe(7)
        fetch_fundamentals_for_universe()

        # Optional: Data validation (soft-fail unless explicitly configured to fail hard)
        if ENABLE_DATA_VALIDATION:
            try:
                from data_validation.gx_checks import validate_recent_daily_bars
                validate_recent_daily_bars(days=5, fail_hard=False)
            except Exception as e:
                log.warning("Data validation skipped or failed softly: %s", e)

        # Phase 2: Features and modeling
        log.info("Phase 2: Feature engineering and modeling")
        build_features()
        _ = train_and_predict_all_models()

        # Phase 3: Trade generation (+ optional sync)
        log.info("Phase 3: Trade generation")
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
    except Exception as e:
        log.exception("Pipeline failed: %s", e)
        return False


if __name__ == "__main__":
    do_sync = os.getenv("SYNC_TO_BROKER", "false").lower() == "true"
    log.info("Starting pipeline (sync_to_broker=%s)", do_sync)
    main(sync_broker=do_sync)
