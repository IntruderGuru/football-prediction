import pandas as pd
from pathlib import Path
import numpy as np


def load_and_merge_fd():
    files = Path("data/raw").glob("fd_*.csv")
    all_df = []

    for file in files:
        df = pd.read_csv(file)

        for col in ["B365H", "B365D", "B365A"]:
            if col not in df.columns:
                df[col] = np.nan

        df["bookie_home"] = pd.to_numeric(df["B365H"], errors="coerce")
        df["bookie_draw"] = pd.to_numeric(df["B365D"], errors="coerce")
        df["bookie_away"] = pd.to_numeric(df["B365A"], errors="coerce")

        df = df.rename(
            columns={
                "Date": "date",
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
                "FTHG": "home_goals",
                "FTAG": "away_goals",
            }
        )

        df["source_file"] = file.name

        all_df.append(
            df[
                [
                    "date",
                    "home_team",
                    "away_team",
                    "home_goals",
                    "away_goals",
                    "bookie_home",
                    "bookie_draw",
                    "bookie_away",
                    "source_file",
                ]
            ]
        )

    final = pd.concat(all_df).sort_values("date")
    return final


if __name__ == "__main__":
    df = load_and_merge_fd()
    df.to_csv("data/raw/football_data_merged.csv", index=False)
    print(f"Merged and saved to data/raw/football_data_merged.csv ({df.shape[0]} rows)")
