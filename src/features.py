import pandas as pd


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the raw model_input DataFrame, compute:
      - xg_diff, goal_diff
      - rolling averages (last 5 matches) of xg_diff, goal_diff
      - form (sum of points last 5 matches)
      - temporal features: day of week, month, days since last match
    """
    df = df.sort_values("date").copy()

    # 1) basic diffs & points
    df["xg_diff"] = df["xG_home"] - df["xG_away"]
    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    df["home_pts"] = df["result"].map({"H": 3, "D": 1, "A": 0})
    df["away_pts"] = df["result"].map({"H": 0, "D": 1, "A": 3})

    window = 5

    # 2) rolling features (last 5 matches per team) — use transform to keep index alignment
    df["home_roll_xg_5"] = df.groupby("home_team")["xg_diff"].transform(
        lambda x: x.shift().rolling(window).mean()
    )
    df["away_roll_xg_5"] = df.groupby("away_team")["xg_diff"].transform(
        lambda x: x.shift().rolling(window).mean()
    )
    df["home_roll_gd_5"] = df.groupby("home_team")["goal_diff"].transform(
        lambda x: x.shift().rolling(window).mean()
    )
    df["away_roll_gd_5"] = df.groupby("away_team")["goal_diff"].transform(
        lambda x: x.shift().rolling(window).mean()
    )
    df["home_roll_form_5"] = df.groupby("home_team")["home_pts"].transform(
        lambda x: x.shift().rolling(window).sum()
    )
    df["away_roll_form_5"] = df.groupby("away_team")["away_pts"].transform(
        lambda x: x.shift().rolling(window).sum()
    )

    # 3) temporal features
    df["dow"] = df["date"].dt.weekday  # 0=Monday…6=Sunday
    df["month"] = df["date"].dt.month

    # 4) days since last match
    df["home_prev_date"] = df.groupby("home_team")["date"].shift(1)
    df["away_prev_date"] = df.groupby("away_team")["date"].shift(1)
    df["home_days_since"] = (df["date"] - df["home_prev_date"]).dt.days
    df["away_days_since"] = (df["date"] - df["away_prev_date"]).dt.days

    # 5) drop intermediate helper cols
    df = df.drop(columns=["home_prev_date", "away_prev_date"])

    return df
