import pandas as pd
import ast


def merge_sources():
    u = pd.read_csv("data/raw/understat_epl.csv")
    f = pd.read_csv("data/raw/football_data_merged.csv")

    f["date"] = pd.to_datetime(f["date"], dayfirst=True)
    u["datetime"] = pd.to_datetime(u["datetime"])
    u = u[u["datetime"].notna()]
    u["date"] = u["datetime"].dt.date
    f["date"] = f["date"].dt.date

    u["h"] = u["h"].apply(ast.literal_eval)
    u["a"] = u["a"].apply(ast.literal_eval)

    # Parse goals and xG from stringified dictionaries
    u["goals"] = u["goals"].apply(ast.literal_eval)
    u["xG"] = u["xG"].apply(ast.literal_eval)

    u["home_goals_understat"] = u["goals"].apply(lambda g: int(g["h"]))
    u["away_goals_understat"] = u["goals"].apply(lambda g: int(g["a"]))

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

    u["h_team"] = u["h"].apply(lambda d: team_map.get(d["title"], d["title"]))
    u["a_team"] = u["a"].apply(lambda d: team_map.get(d["title"], d["title"]))

    merged = pd.merge(
        f,
        u,
        left_on=["date", "home_team", "away_team"],
        right_on=["date", "h_team", "a_team"],
        how="inner",
    )

    print(f"Merged shape: {merged.shape}")
    merged[["date", "home_team", "away_team", "bookie_home", "xG"]].head()

    df = pd.DataFrame(
        {
            "date": merged["date"],
            "home_team": merged["home_team"],
            "away_team": merged["away_team"],
            "home_goals": merged["home_goals"],
            "away_goals": merged["away_goals"],
            "xG_home": merged["xG_home"],
            "xG_away": merged["xG_away"],
            "bookie_home": merged["bookie_home"],
            "bookie_draw": merged["bookie_draw"],
            "bookie_away": merged["bookie_away"],
        }
    )

    df["date"] = pd.to_datetime(df["date"])

    df[["bookie_home", "bookie_draw", "bookie_away"]] = df[
        ["bookie_home", "bookie_draw", "bookie_away"]
    ].astype(float)

    def get_result(row):
        if row["home_goals"] > row["away_goals"]:
            return "H"
        elif row["home_goals"] < row["away_goals"]:
            return "A"
        else:
            return "D"

    df["result"] = df.apply(get_result, axis=1)

    df["bookie_sum"] = (
        1 / df["bookie_home"] + 1 / df["bookie_draw"] + 1 / df["bookie_away"]
    )
    df["bookie_prob_home"] = (1 / df["bookie_home"]) / df["bookie_sum"]
    df["bookie_prob_draw"] = (1 / df["bookie_draw"]) / df["bookie_sum"]
    df["bookie_prob_away"] = (1 / df["bookie_away"]) / df["bookie_sum"]

    df.to_parquet("data/processed/merged.parquet", index=False, engine="fastparquet")

    print("Merged file saved.")


if __name__ == "__main__":
    try:
        merge_sources()
    except Exception as e:
        print(f"Error during merge: {e}")
