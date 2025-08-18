# main.py
# This file contains the main logic for the Google Cloud Function.
# It is responsible for ingesting daily stock data from Alpaca
# and publishing it to a Google Cloud Pub/Sub topic.

import base64
import json
import os
from datetime import datetime, timedelta

from alpaca_trade_api.rest import REST, APIError
from google.cloud import pubsub_v1, secretmanager

# --- Configuration ---
# GCP Project ID and Pub/Sub topic name are retrieved from environment variables.
# These should be set during the deployment of the Cloud Function.
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
PUB_SUB_TOPIC = os.environ.get('PUB_SUB_TOPIC', 'daily-bars-raw')

# The name of the secret in Google Secret Manager containing Alpaca API keys.
# Format: projects/your-project-number/secrets/your-secret-name/versions/latest
ALPACA_SECRET_NAME = os.environ.get('ALPACA_SECRET_NAME')

# Define the universe of small-cap stocks to track.
# In a production system, this list would be dynamically loaded from a database (e.g., the Security Master in Cloud SQL).
# For this version, we use a static list for simplicity.
STOCK_UNIVERSE = [
    'SMCI', 'CRWD', 'DDOG', 'MDB', 'OKTA', 'PLTR', 'SNOW', 'ZS',
    'ETSY', 'PINS', 'ROKU', 'SQ', 'TDOC', 'TWLO', 'U', 'ZM'
]

# --- Initialize Clients (Global Scope) ---
# It's a best practice in Cloud Functions to initialize clients outside the main function handler.
# This allows for connection reuse between invocations, improving performance.

def get_alpaca_api_client():
    """
    Retrieves Alpaca API keys from Secret Manager and returns an authenticated Alpaca API client.
    """
    try:
        # Create the Secret Manager client.
        secret_client = secretmanager.SecretManagerServiceClient()

        # Access the secret version.
        response = secret_client.access_secret_version(request={"name": ALPACA_SECRET_NAME})
        payload = response.payload.data.decode("UTF-8")
        secrets = json.loads(payload)

        # Create and return the Alpaca REST client.
        api = REST(
            key_id=secrets['ALPACA_API_KEY'],
            secret_key=secrets['ALPACA_SECRET_KEY'],
            base_url=secrets.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets') # Default to paper trading
        )
        print("Successfully created Alpaca API client.")
        return api
    except Exception as e:
        print(f"Error creating Alpaca API client: {e}")
        raise

# Initialize clients in the global scope.
publisher = pubsub_v1.PublisherClient()
api = get_alpaca_api_client()
topic_path = publisher.topic_path(GCP_PROJECT_ID, PUB_SUB_TOPIC)


def ingest_daily_data(event, context):
    """
    Google Cloud Function entry point.
    This function is triggered by an event (e.g., a Pub/Sub message from Cloud Scheduler).
    It fetches daily bar data for the defined stock universe and publishes it.

    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    print("Starting daily data ingestion process...")

    # Calculate the date for which to fetch data (yesterday).
    # This assumes the function runs after market close.
    trade_date = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')

    try:
        # Fetch the bar data from Alpaca.
        # The API call is vectorized, requesting data for all symbols at once.
        barset = api.get_bars(
            STOCK_UNIVERSE,
            '1Day',
            start=trade_date,
            end=trade_date
        ).df

        if barset.empty:
            print(f"No data returned from Alpaca for date: {trade_date}. This may be a weekend or holiday.")
            return 'No data for the requested date.'

        print(f"Successfully retrieved {len(barset)} bars from Alpaca.")

        # Process and publish a message for each stock bar.
        messages_published = 0
        for symbol, row in barset.iterrows():
            message_payload = {
                'symbol': symbol,
                'timestamp': row.name.isoformat(), # The index is the timestamp
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume']),
                'trade_count': int(row.get('trade_count', 0)), # Not always present
                'vwap': float(row.get('vwap', 0.0)) # Not always present
            }

            # Data must be a bytestring for Pub/Sub.
            data = json.dumps(message_payload).encode('utf-8')

            # Publish the message. The publisher client handles batching automatically.
            future = publisher.publish(topic_path, data)
            future.result() # Wait for the publish operation to complete.
            messages_published += 1

        print(f"Successfully published {messages_published} messages to topic '{PUB_SUB_TOPIC}'.")
        return f"Ingestion complete. Published {messages_published} messages."

    except APIError as e:
        print(f"Alpaca API Error: {e}")
        # Depending on the error, you might want to raise it to trigger a retry.
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


