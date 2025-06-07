import pandas as pd
import ast
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa


def merge_sources() -> None:
    u = pd.read_csv("data/raw/understat_all.csv")
    f = pd.read_csv("data/raw/football_data_merged.csv")

    f["date"] = pd.to_datetime(f["date"], dayfirst=False).dt.normalize()
    u["datetime"] = pd.to_datetime(u["datetime"])
    u = u[u["datetime"].notna()]
    u["date"] = u["datetime"].dt.normalize()

    for col in ["h", "a", "goals", "xG"]:
        u[col] = u[col].apply(ast.literal_eval)

    u["home_goals"] = u["goals"].apply(lambda g: int(g["h"]))
    u["away_goals"] = u["goals"].apply(lambda g: int(g["a"]))
    u["xG_home"] = u["xG"].apply(lambda g: float(g["h"]))
    u["xG_away"] = u["xG"].apply(lambda g: float(g["a"]))

    team_map = {
        "Manchester United": "Man United",
        "Manchester City": "Man City",
        "Wolverhampton Wanderers": "Wolves",
        "Brighton & Hove Albion": "Brighton",
        "Tottenham Hotspur": "Tottenham",
        "West Ham United": "West Ham",
        "Newcastle United": "Newcastle",
        "Nottingham Forest": "Nott'm Forest",
        "Sheffield United": "Sheffield Utd",
        "Leeds United": "Leeds",
        "Leicester City": "Leicester",
        "Aston Villa": "Aston Villa",
        "Crystal Palace": "Crystal Palace",
        "Everton": "Everton",
        "Arsenal": "Arsenal",
        "Liverpool": "Liverpool",
        "Chelsea": "Chelsea",
        "Southampton": "Southampton",
        "Burnley": "Burnley",
        "Bournemouth": "Bournemouth",
        "Watford": "Watford",
        "Brentford": "Brentford",
        "Fulham": "Fulham",
    }
    u["home_team"] = u["h"].apply(lambda d: team_map.get(d["title"], d["title"]))
    u["away_team"] = u["a"].apply(lambda d: team_map.get(d["title"], d["title"]))

    merged = f.merge(
        u[["date", "league", "home_team", "away_team", "xG_home", "xG_away"]],
        on=["date", "league", "home_team", "away_team"],
        how="left",
    )

    merged["bookie_sum"] = (
        1 / merged["bookie_home"]
        + 1 / merged["bookie_draw"]
        + 1 / merged["bookie_away"]
    )
    merged["bookie_prob_home"] = (1 / merged["bookie_home"]) / merged["bookie_sum"]
    merged["bookie_prob_draw"] = (1 / merged["bookie_draw"]) / merged["bookie_sum"]
    merged["bookie_prob_away"] = (1 / merged["bookie_away"]) / merged["bookie_sum"]

    merged["result"] = np.select(
        [
            merged["home_goals"] > merged["away_goals"],
            merged["home_goals"] < merged["away_goals"],
        ],
        ["H", "A"],
        default="D",
    )

    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    num_cols = [
        "home_goals",
        "away_goals",
        "xG_home",
        "xG_away",
        "bookie_home",
        "bookie_draw",
        "bookie_away",
        "bookie_prob_home",
        "bookie_prob_draw",
        "bookie_prob_away",
    ]
    merged[num_cols] = merged[num_cols].apply(pd.to_numeric, errors="coerce")

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(merged)
    pq.write_table(table, "data/processed/merged.parquet")
    print(f" merged.parquet saved  ({merged.shape[0]} rows)")


if __name__ == "__main__":
    merge_sources()
