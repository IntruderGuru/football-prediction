import pandas as pd, ast, numpy as np
from pathlib import Path


def merge_sources():
    u = pd.read_csv("data/raw/understat_all.csv")
    f = pd.read_csv("data/raw/football_data_merged.csv")
    f["date"] = pd.to_datetime(f["date"])
    u["datetime"] = pd.to_datetime(u["datetime"])
    u = u[u["datetime"].notna()]
    u["date"] = u["datetime"].dt.normalize()
    u["h"] = u["h"].apply(ast.literal_eval)
    u["a"] = u["a"].apply(ast.literal_eval)
    u["goals"] = u["goals"].apply(ast.literal_eval)
    u["xG"] = u["xG"].apply(ast.literal_eval)
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
    merged["date"] = pd.to_datetime(merged["date"])
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    merged.to_parquet(
        "data/processed/merged.parquet", index=False, engine="fastparquet"
    )


if __name__ == "__main__":
    merge_sources()
