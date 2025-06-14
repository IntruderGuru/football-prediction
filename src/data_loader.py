import pandas as pd


def load_data(path: str = "data/processed/model_input.parquet") -> pd.DataFrame:
    
    return pd.read_parquet(path)