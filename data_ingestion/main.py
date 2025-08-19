# main.py
# This file contains the main logic for the Google Cloud Function.
# It is responsible for ingesting daily stock data from Alpaca
# and publishing it to a Google Cloud Pub/Sub topic.
# This version is updated to perform a historical backfill.

import base64
import json
import os
from datetime import datetime, timedelta
import time

from alpaca_trade_api.rest import REST, APIError
from google.cloud import pubsub_v1, secretmanager
# from tqdm import tqdm # DEBUG: Temporarily removed to isolate the deployment crash.

# --- Configuration ---
# These are retrieved when the function is invoked, not at deploy time.
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
PUB_SUB_TOPIC = os.environ.get('PUB_SUB_TOPIC', 'daily-bars-raw')
ALPACA_SECRET_NAME = os.environ.get('ALPACA_SECRET_NAME')

STOCK_UNIVERSE = [
    'SMCI', 'CRWD', 'DDOG', 'MDB', 'OKTA', 'PLTR', 'SNOW', 'ZS',
    'ETSY', 'PINS', 'ROKU', 'SQ', 'TDOC', 'TWLO', 'U', 'ZM'
]

# --- Client Initialization (Lazy) ---
# Declare clients globally but initialize them inside the function handler
# to ensure environment variables are available.
api = None
publisher = None
topic_path = None

def get_alpaca_api_client():
    """
    Retrieves Alpaca API keys from Secret Manager and returns an authenticated Alpaca API client.
    """
    try:
        secret_client = secretmanager.SecretManagerServiceClient()
        response = secret_client.access_secret_version(request={"name": ALPACA_SECRET_NAME})
        payload = response.payload.data.decode("UTF-8")
        secrets = json.loads(payload)
        
        client = REST(
            key_id=secrets['ALPACA_API_KEY'],
            secret_key=secrets['ALPACA_SECRET_KEY'],
            base_url=secrets.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        )
        print("Successfully created Alpaca API client.")
        return client
    except Exception as e:
        print(f"Error creating Alpaca API client: {e}")
        raise

def ingest_daily_data(event, context):
    """
    Google Cloud Function entry point.
    This function now performs a historical backfill for the last 2 years.
    """
    global api, publisher, topic_path

    # LAZY INITIALIZATION: Initialize clients on first invocation
    if not api:
        api = get_alpaca_api_client()
    if not publisher:
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(GCP_PROJECT_ID, PUB_SUB_TOPIC)

    print("Starting historical data backfill process...")

    # Define the time range for the backfill
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365 * 2)

    # Process symbols in chunks to respect API rate limits
    chunk_size = 100
    total_messages_published = 0

    # DEBUG: Temporarily removed tqdm wrapper to isolate the deployment crash.
    for i in range(0, len(STOCK_UNIVERSE), chunk_size):
        chunk = STOCK_UNIVERSE[i:i + chunk_size]
        print(f"Processing symbol chunk starting with {chunk[0]}...")
        try:
            barset = api.get_bars(
                chunk,
                '1Day',
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            ).df

            if barset.empty:
                print(f"No data returned from Alpaca for chunk starting with {chunk[0]}.")
                continue

            print(f"Retrieved {len(barset)} bars for chunk starting with {chunk[0]}.")

            for symbol, row in barset.iterrows():
                # The symbol is part of the multi-index, extract it
                actual_symbol = symbol[0]
                
                message_payload = {
                    'symbol': actual_symbol,
                    'timestamp': row.name.isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume']),
                    'trade_count': int(row.get('trade_count', 0)),
                    'vwap': float(row.get('vwap', 0.0))
                }

                data = json.dumps(message_payload).encode('utf-8')
                future = publisher.publish(topic_path, data)
                future.result()
                total_messages_published += 1

        except APIError as e:
            print(f"Alpaca API Error for chunk {chunk}: {e}")
            time.sleep(30) # Wait before retrying next chunk
        except Exception as e:
            print(f"An unexpected error occurred for chunk {chunk}: {e}")
            time.sleep(30)

    print(f"Backfill complete. Published {total_messages_published} messages to topic '{PUB_SUB_TOPIC}'.")
    return f"Backfill complete. Published {total_messages_published} messages."
