# main.py
# Updated to use the modern alpaca-py library and connect to a database.

import os
import json
from datetime import datetime, timedelta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from sqlalchemy import create_engine, text
from flask import Flask, request

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration ---
# On Render, environment variables are set in the dashboard.
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///data_ingestion.db")

SYMBOLS = [
    "SMCI",
    "CRWD",
    "DDOG",
    "MDB",
    "OKTA",
    "PLTR",
    "SNOW",
    "ZS",
    "ETSY",
    "PINS",
    "ROKU",
    "SQ",
    "TDOC",
    "TWLO",
    "U",
    "ZM",
]

# --- Client Initialization ---
# Clients are initialized once when the application starts.
db_engine = create_engine(DATABASE_URL)
alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def fetch_and_store(symbols, start_date, end_date):
    """Fetch bars from Alpaca and persist them to the database.

    Returns the number of rows written.
    """
    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
    )

    bars = alpaca_client.get_stock_bars(request_params)
    bars_df = bars.df

    if bars_df.empty:
        return 0

    # The new library returns a multi-index DataFrame. Reset index to make 'symbol' a column.
    bars_df.reset_index(inplace=True)

    # Rename columns to match your database schema
    bars_df.rename(
        columns={"timestamp": "timestamp", "trade_count": "trade_count", "vwap": "vwap"},
        inplace=True,
    )

    # Save data to the PostgreSQL database
    with db_engine.connect() as connection:
        # Using a temporary table for a safe bulk insert
        bars_df.to_sql("daily_bars_temp", connection, if_exists="replace", index=False)

        # Upsert from the temporary table into the main table
        upsert_query = text(
            """
            INSERT INTO daily_bars (symbol, timestamp, open, high, low, close, volume, trade_count, vwap)
            SELECT symbol, timestamp, open, high, low, close, volume, trade_count, vwap FROM daily_bars_temp
            ON CONFLICT (symbol, timestamp) DO NOTHING;
            """
        )
        connection.execute(upsert_query)
        connection.commit()  # Commit the transaction

        # Drop the temporary table
        connection.execute(text("DROP TABLE daily_bars_temp;"))
        connection.commit()

    return len(bars_df)


@app.route('/ingest', methods=['POST'])
def ingest_daily_data():
    """Flask route to trigger the historical data backfill process."""
    print("Starting historical data backfill process...")

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365 * 2)

    try:
        written = fetch_and_store(SYMBOLS, start_date, end_date)
        if written == 0:
            message = "No data returned from Alpaca."
            print(message)
            return message, 200

        result_message = f"Backfill complete. Saved {written} bars to the database."
        print(result_message)
        return result_message, 200

    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(error_message)
        return error_message, 500

# Health check route
@app.route('/')
def health_check():
    return "Service is running.", 200

if __name__ == '__main__':
    # This part is for local testing and is not used by gunicorn on Render.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))


