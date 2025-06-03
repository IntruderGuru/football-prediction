import pandas as pd, numpy as np, re, itertools
from pathlib import Path

DIV_MAP = {
    "E0": "EPL",
    "SP1": "LALIGA",
    "D1": "BUNDESLIGA",
    "I1": "SERIEA",
    "F1": "LIGUE1",
}
RAW = Path("data/raw")


def load_fd() -> pd.DataFrame:
    rows = []
    for fp in RAW.glob("fd_*.csv"):
        m = re.match(r"fd_(?P<div>[A-Z0-9]+)_(?P<season>\d{4}).csv", fp.name)
        if not m:
            continue
        league = DIV_MAP[m["div"]]
        df = pd.read_csv(
            fp,
            usecols=[
                "Date",
                "HomeTeam",
                "AwayTeam",
                "FTHG",
                "FTAG",
                "B365H",
                "B365D",
                "B365A",
            ],
        )
        df = df.rename(
            columns={
                "Date": "date",
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
                "FTHG": "home_goals",
                "FTAG": "away_goals",
                "B365H": "bookie_home",
                "B365D": "bookie_draw",
                "B365A": "bookie_away",
            }
        )
        df["league"] = league
        df["date"] = pd.to_datetime(df["date"], dayfirst=True).dt.date
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    fd_df = load_fd()
    fd_df.to_csv("data/raw/football_data_merged.csv", index=False)


if __name__ == "__main__":
    main()
