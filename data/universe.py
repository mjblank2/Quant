from __future__ import annotations
import logging
import pandas as pd

log = logging.getLogger("data.universe")


def _list_alpaca_assets() -> pd.DataFrame:
    """Get list of assets from Alpaca API."""
    try:
        from config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL
        if not APCA_API_KEY_ID:
            log.warning("No Alpaca API key configured, returning empty universe")
            return pd.DataFrame(columns=['symbol', 'name', 'exchange'])

        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(key_id=APCA_API_KEY_ID, secret_key=APCA_API_SECRET_KEY, base_url=APCA_API_BASE_URL)

        # Get tradable assets
        assets = api.list_assets(status='active', asset_class='us_equity')

        data = []
        for asset in assets:
            if asset.tradable and asset.shortable:  # Focus on tradable assets
                data.append({
                    'symbol': asset.symbol,
                    'name': getattr(asset, 'name', asset.symbol),
                    'exchange': getattr(asset, 'exchange', 'UNKNOWN')
                })

        df = pd.DataFrame(data)
        log.info(f"Retrieved {len(df)} tradable assets from Alpaca")
        return df

    except Exception as e:
        log.error(f"Failed to list Alpaca assets: {e}", exc_info=True)
        return pd.DataFrame(columns=['symbol', 'name', 'exchange'])


def rebuild_universe() -> pd.DataFrame:
    """
    Rebuild the universe of stocks by fetching from Alpaca and updating the database.

    Returns:
        DataFrame with universe data containing columns: symbol, name, exchange
    """
    log.info("Starting universe rebuild")

    try:
        from db import upsert_dataframe, Universe
        from datetime import datetime

        # Get assets from Alpaca
        universe_df = _list_alpaca_assets()

        if universe_df.empty:
            log.warning("No assets retrieved, universe remains empty")
            return universe_df

        # Add default values for Universe table schema
        universe_df['market_cap'] = None
        universe_df['adv_usd_20'] = None
        universe_df['included'] = True
        universe_df['last_updated'] = datetime.utcnow()

        # Filter to reasonable universe size (focus on liquid stocks)
        # This is a placeholder filter - in production you'd add market cap, volume filters
        universe_df = universe_df.head(2000)  # Limit for demo

        log.info(f"Upserting {len(universe_df)} symbols to universe table")

        # Upsert to database
        upsert_dataframe(universe_df, Universe, ['symbol'])

        log.info(f"Universe rebuild completed with {len(universe_df)} symbols")
        return universe_df[['symbol', 'name', 'exchange']]  # Return basic columns

    except Exception as e:
        log.error(f"Universe rebuild failed: {e}", exc_info=True)
        # Return empty DataFrame on error
        return pd.DataFrame(columns=['symbol', 'name', 'exchange'])
