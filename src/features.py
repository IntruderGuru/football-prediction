import pandas as pd
import numpy as np

K_ELO = 40  # Elo learning rate
WINDOW = 5  # rolling window size
EWM_SPAN = 10  # span for exponential smoothing


def _rolling_slope(series: pd.Series) -> float:
    """Return linear trend slope over the last WINDOW points."""
    if series.isna().any() or len(series) < WINDOW:
        return np.nan
    y = series.values
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    return slope


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extended feature engineering without data leakage:
      - Date features
      - Rolling and EWM xG metrics
      - Rolling goal difference and form
      - Exponential moving average of xG
      - Elo ratings and trend
      - Poisson-based scoring rates
      - Expanding goal averages
      - Engineered interaction features
    """
    df = df.sort_values("date").copy()

    # 1. Date features
    df["month"] = df["date"].dt.month

    # 2. Rolling xG for and against
    df["xG_home_roll"] = df.groupby("home_team")["xG_home"].transform(
        lambda s: s.shift().rolling(WINDOW).mean()
    )
    df["xG_away_roll"] = df.groupby("away_team")["xG_away"].transform(
        lambda s: s.shift().rolling(WINDOW).mean()
    )
    df["home_roll_xg_against"] = df.groupby("home_team")["xG_away"].transform(
        lambda s: s.shift().rolling(WINDOW).mean()
    )
    df["away_roll_xg_against"] = df.groupby("away_team")["xG_home"].transform(
        lambda s: s.shift().rolling(WINDOW).mean()
    )

    # 2a. Exponential weighted xG
    df["xG_home_ewm"] = df.groupby("home_team")["xG_home"].transform(
        lambda s: s.shift().ewm(span=EWM_SPAN, adjust=False).mean()
    )
    df["xG_away_ewm"] = df.groupby("away_team")["xG_away"].transform(
        lambda s: s.shift().ewm(span=EWM_SPAN, adjust=False).mean()
    )

    # 3. Rolling goal difference
    gd = df["home_goals"] - df["away_goals"]
    df["home_roll_gd"] = (
        df.assign(gd=gd)
        .groupby("home_team")["gd"]
        .transform(lambda s: s.shift().rolling(WINDOW).mean())
    )
    df["away_roll_gd"] = (
        df.assign(gd=gd)
        .groupby("away_team")["gd"]
        .transform(lambda s: s.shift().rolling(WINDOW).mean())
    )

    # 4. Rolling form (points)
    pts = df["result"].map({"H": 3, "D": 1, "A": 0})
    df["home_roll_pts"] = (
        df.assign(pts=pts)
        .groupby("home_team")["pts"]
        .transform(lambda s: s.shift().rolling(WINDOW).sum())
    )
    df["away_roll_pts"] = (
        df.assign(pts=pts)
        .groupby("away_team")["pts"]
        .transform(lambda s: s.shift().rolling(WINDOW).sum())
    )

    # 4a. Form trend (slope)
    df["form_slope_home"] = (
        df.assign(pts=pts)
        .groupby("home_team")["pts"]
        .transform(lambda s: s.shift().rolling(WINDOW).apply(_rolling_slope, raw=False))
    )
    df["form_slope_away"] = (
        df.assign(pts=pts)
        .groupby("away_team")["pts"]
        .transform(lambda s: s.shift().rolling(WINDOW).apply(_rolling_slope, raw=False))
    )

    # 5. Days since last match
    df["home_prev"] = df.groupby("home_team")["date"].shift(1)
    df["away_prev"] = df.groupby("away_team")["date"].shift(1)
    df["home_days_since"] = (df["date"] - df["home_prev"]).dt.days
    df["away_days_since"] = (df["date"] - df["away_prev"]).dt.days

    # 6. Elo ratings
    elo = {}
    home_elos, away_elos = [], []
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        elo.setdefault(h, 1500.0)
        elo.setdefault(a, 1500.0)
        home_elos.append(elo[h])
        away_elos.append(elo[a])
        score_h = 1.0 if row["result"] == "H" else 0.5 if row["result"] == "D" else 0.0
        exp_h = 1 / (1 + 10 ** ((elo[a] - elo[h]) / 400))
        elo[h] += K_ELO * (score_h - exp_h)
        elo[a] += K_ELO * ((1 - score_h) - (1 - exp_h))
    df["elo_home"] = home_elos
    df["elo_away"] = away_elos

    # 6a. Elo change over WINDOW matches
    df["elo_change_home"] = df.groupby("home_team")["elo_home"].transform(
        lambda s: s.shift().diff(periods=WINDOW)
    )
    df["elo_change_away"] = df.groupby("away_team")["elo_away"].transform(
        lambda s: s.shift().diff(periods=WINDOW)
    )

    # 7. Poisson-based scoring rates
    df["lambda_home_for"] = df.groupby("home_team")["home_goals"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df["lambda_home_against"] = df.groupby("home_team")["away_goals"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df["lambda_away_for"] = df.groupby("away_team")["away_goals"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df["lambda_away_against"] = df.groupby("away_team")["home_goals"].transform(
        lambda s: s.shift().expanding().mean()
    )

    # 7a. Expanding goal averages
    df["avg_goals_home"] = df.groupby("home_team")["home_goals"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df["avg_goals_away"] = df.groupby("away_team")["away_goals"].transform(
        lambda s: s.shift().expanding().mean()
    )

    # 8. Engineered interaction features
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["xg_diff_roll"] = df["xG_home_roll"] - df["away_roll_xg_against"]
    df["form_momentum"] = df["home_roll_pts"] - df["away_roll_pts"]
    df["lambda_attack_ratio"] = df["lambda_home_for"] / (
        df["lambda_away_against"] + 1e-6
    )
    df["bookie_form_interact"] = df["bookie_prob_home"] * df["form_momentum"]
    df["bookie_rest_draw"] = df["bookie_prob_draw"] * (
        df["home_days_since"] - df["away_days_since"]
    )
    df["elo_xg_ratio"] = df["elo_diff"] / (df["xg_diff_roll"].abs() + 1e-6)
    df["home_xg_std"] = df.groupby("home_team")["xG_home"].transform(
        lambda s: s.shift().rolling(WINDOW).std()
    )
    df["away_xg_std"] = df.groupby("away_team")["xG_away"].transform(
        lambda s: s.shift().rolling(WINDOW).std()
    )

    # additional interactions
    df["xg_elo_interact"] = df["xg_diff_roll"] * df["elo_diff"]
    close_match = (df["elo_diff"].abs() < 25).astype(int)
    df["bookie_draw_balance"] = df["bookie_prob_draw"] * close_match
    df["form_rest_interact"] = df["form_momentum"] * (
        df["home_days_since"] - df["away_days_since"]
    )

    # remove temp columns
    df.drop(["home_prev", "away_prev"], axis=1, inplace=True)
    return df
