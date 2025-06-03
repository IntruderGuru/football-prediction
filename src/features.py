import pandas as pd


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    df["xg_diff"] = df["xG_home"] - df["xG_away"]

    required = ["bookie_home", "bookie_draw", "bookie_away"]
    if all(col in df.columns for col in required):
        df["bookie_sum"] = (
            1 / df["bookie_home"] + 1 / df["bookie_draw"] + 1 / df["bookie_away"]
        )
        df["bookie_prob_home"] = (1 / df["bookie_home"]) / df["bookie_sum"]
        df["bookie_prob_draw"] = (1 / df["bookie_draw"]) / df["bookie_sum"]
        df["bookie_prob_away"] = (1 / df["bookie_away"]) / df["bookie_sum"]

    def get_result(row):
        if row["home_goals"] > row["away_goals"]:
            return "H"
        elif row["home_goals"] < row["away_goals"]:
            return "A"
        else:
            return "D"

    df["result"] = df.apply(get_result, axis=1)

    return df
