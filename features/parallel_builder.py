import dask.dataframe as dd
import pandas as pd
import multiprocessing


def compute_asset_features(df_asset: pd.DataFrame) -> pd.DataFrame:
    """
    Example feature engineering function applied per asset.
    This function should compute technical indicators or other features for a single asset.
    """
    df_asset = df_asset.copy()
    # Example: calculate a 20-period simple moving average on the 'Close' column if present
    if 'Close' in df_asset.columns:
        df_asset['SMA_20'] = df_asset['Close'].rolling(window=20).mean()
    return df_asset


def build_features_parallel(df_long: pd.DataFrame, n_partitions: int | None = None) -> pd.DataFrame:
    """
    Applies feature engineering in parallel using Dask.

    Parameters
    ----------
    df_long : pd.DataFrame
        DataFrame in long format with an 'Asset' or 'symbol' column and a datetime index.
    n_partitions : int, optional
        Number of partitions for Dask. Defaults to the number of CPU cores.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features computed in parallel per asset.
    """
    if n_partitions is None:
        n_partitions = multiprocessing.cpu_count()

    # Convert pandas DataFrame to a Dask DataFrame
    ddf = dd.from_pandas(df_long, npartitions=n_partitions)

    # Use a small sample to infer metadata for the output of compute_asset_features
    sample = df_long.head(1)
    meta = compute_asset_features(sample)

    # Determine the grouping key
    group_key = 'Asset' if 'Asset' in df_long.columns else 'symbol'

    # Apply feature computation in parallel by group
    result_ddf = ddf.groupby(group_key).apply(compute_asset_features, meta=meta)

    # Compute the result back to a pandas DataFrame
    result_df = result_ddf.compute()
    return result_df
