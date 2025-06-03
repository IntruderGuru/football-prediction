import pandas as pd


def load_data(path="data/processed/merged.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)
