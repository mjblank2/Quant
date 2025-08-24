from __future__ import annotations
import pandas as pd
from db import upsert_dataframe, AltSignal

def load_alt_signals_from_csv(csv_path: str, name: str, ts_col: str = "ts", symbol_col: str = "symbol", value_col: str = "value"):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={ts_col: "ts", symbol_col: "symbol", value_col: "value"})
    df["ts"] = pd.to_datetime(df["ts"]).dt.date
    df["name"] = name
    upsert_dataframe(df[["symbol","ts","name","value"]], AltSignal, ["symbol","ts","name"])

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    p.add_argument("name")
    args = p.parse_args()
    load_alt_signals_from_csv(args.csv_path, args.name)
