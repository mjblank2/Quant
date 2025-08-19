# main.py
# This file contains the main logic for the data ingestion service,
# adapted to run as a web service on Render.

import os
import json
from datetime import datetime, timedelta
import time

from alpaca_trade_api.rest import REST, APIError
from google.cloud import pubsub_v1
from google.oauth2 import service_account
from flask import Flask, request

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration ---
# On Render, environment variables are set in the dashboard.
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
PUB_SUB_TOPIC = os.environ.get('PUB_SUB_TOPIC', 'daily-bars-raw')

# Load BigQuery credentials from the environment variable
# This is a secure way to handle the JSON keyfile on Render.
credentials_json_str = os.environ.get('BIGQUERY_CREDENTIALS_JSON')
credentials_info = json.loads(credentials_json_str)
credentials = service_account.Credentials.from_service_account_info(credentials_info)

STOCK_UNIVERSE = [
    'SMCI', 'CRWD', 'DDOG', 'MDB', 'OKTA', 'PLTR', 'SNOW', 'ZS',
    'ETSY', 'PINS', 'ROKU', 'SQ', 'TDOC', 'TWLO', 'U', 'ZM'
]

# --- Client Initialization ---
# Clients are initialized once when the application starts.
api = REST(key_id=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets')
publisher = pubsub_v1.PublisherClient(credentials=credentials)
topic_path = publisher.topic_path(GCP_PROJECT_ID, PUB_SUB_TOPIC)


@app.route('/ingest', methods=['POST'])
def ingest_daily_data():
    """
    Flask route to trigger the historical data backfill process.
    This is called by the Render Cron Job.
    """
    print("Starting historical data backfill process...")

    # Define the time range for the backfill
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365 * 2)

    # Process symbols in chunks to respect API rate limits
    chunk_size = 100
    total_messages_published = 0

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
            time.sleep(30)
        except Exception as e:
            print(f"An unexpected error occurred for chunk {chunk}: {e}")
            time.sleep(30)

    result_message = f"Backfill complete. Published {total_messages_published} messages to topic '{PUB_SUB_TOPIC}'."
    print(result_message)
    return result_message, 200

# Health check route
@app.route('/')
def health_check():
    return "Service is running.", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

