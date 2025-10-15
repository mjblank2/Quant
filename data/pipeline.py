import pandas as pd
import numpy as np

def automated_data_quality_checks(df, z_score_threshold=5):
    """
    Performs automated data quality checks and cleaning.
    """
    print("Running data quality checks...")

    # 1. Missing Values Handling (Time-series specific: ffill then bfill)
    df = df.ffill().bfill()

    # 2. Outlier Detection and Treatment (Winsorization)
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        mean = df[numeric_cols].mean()
        std_dev = df[numeric_cols].std()
        
        # Handle constant columns (std_dev == 0) safely
        std_dev_safe = std_dev.replace(0, 1e-9)
        
        # Calculate bounds for Winsorization
        upper_bound = mean + z_score_threshold * std_dev
        lower_bound = mean - z_score_threshold * std_dev
        
        # Apply clipping (Winsorization)
        df[numeric_cols] = df[numeric_cols].clip(lower=lower_bound, upper=upper_bound, axis=1)

    # 3. Infinite Values Handling
    if df[numeric_cols].isin([np.inf, -np.inf]).any().any():
        print("Warning: Infinite values detected. Replacing with NaN and filling.")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()

    print("Data quality checks completed.")
    return df

def save_to_parquet(df, filepath):
    """Saves a DataFrame to Parquet format (requires pyarrow)."""
    try:
        # Use snappy compression and pyarrow engine for efficiency
        df.to_parquet(filepath, engine='pyarrow', compression='snappy', index=True)
        print(f"Data saved efficiently to {filepath}")
    except ImportError:
        print("Error: pyarrow not installed. Install with 'pip install pyarrow'.")
    except Exception as e:
        print(f"Error saving to Parquet: {e}")
