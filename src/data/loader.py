from pathlib import Path
from datetime import date
import pandas as pd


def clip_season(
    df: pd.DataFrame, cutoff: pd.Timestamp = pd.Timestamp("2025-05-15")
) -> pd.DataFrame:
    return df[df["date"] <= cutoff].copy()


class FootDataLoader:
    def __init__(self, cache_path: Path = Path("data/processed/model_input.parquet")):
        self.cache_path = cache_path

    def build(self, force: bool = False):
        if force or not self.cache_path.exists():
            raise RuntimeError(
                "Brak gotowego pliku model_input.parquet – uruchom scripts/merge_sources.py"
            )

    def get_training_data(self, force: bool = False) -> pd.DataFrame:
        self.build(force)
        df = pd.read_parquet(self.cache_path)
        return clip_season(df)
