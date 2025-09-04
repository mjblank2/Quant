from __future__ import annotations
import logging
import pandas as pd

log = logging.getLogger("data.universe")

def _list_alpaca_assets() -> pd.DataFrame:
    try:
        # placeholder implementation
        return pd.DataFrame(columns=['symbol','name','exchange'])
    except Exception as e:
        log.error(f"Failed to list Alpaca assets: {e}", exc_info=True)
        return pd.DataFrame(columns=['symbol','name','exchange'])
