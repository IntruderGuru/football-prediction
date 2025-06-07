import pandas as pd


def load_data(path: str = "data/processed/model_input.parquet") -> pd.DataFrame:
    """
    Loads the pre-filtered model input (only rows with complete xG + bookie odds).
    """
    return pd.read_parquet(path)
