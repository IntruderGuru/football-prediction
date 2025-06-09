import pandas as pd

K_ELO = 20  # szybkość uczenia rankingów Elo
WINDOW = 5  # rozmiar bufora rolling


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Wylicza wszystkie cechy potrzebne modelowi v2."""

    df = df.sort_values("date").copy()

    # 1️⃣  Podstawowe różnice & punkty
    df["xg_diff"] = df["xG_home"] - df["xG_away"]
    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    df["home_pts"] = df["result"].map({"H": 3, "D": 1, "A": 0})
    df["away_pts"] = df["result"].map({"H": 0, "D": 1, "A": 3})

    # 2️⃣  Rolling — ostatnie 5 meczów
    df["home_roll_xg_5"] = df.groupby("home_team")["xg_diff"].transform(
        lambda s: s.shift().rolling(WINDOW).mean()
    )
    df["away_roll_xg_5"] = df.groupby("away_team")["xg_diff"].transform(
        lambda s: s.shift().rolling(WINDOW).mean()
    )
    df["home_roll_gd_5"] = df.groupby("home_team")["goal_diff"].transform(
        lambda s: s.shift().rolling(WINDOW).mean()
    )
    df["away_roll_gd_5"] = df.groupby("away_team")["goal_diff"].transform(
        lambda s: s.shift().rolling(WINDOW).mean()
    )
    df["home_roll_form_5"] = df.groupby("home_team")["home_pts"].transform(
        lambda s: s.shift().rolling(WINDOW).sum()
    )
    df["away_roll_form_5"] = df.groupby("away_team")["away_pts"].transform(
        lambda s: s.shift().rolling(WINDOW).sum()
    )

    # 3️⃣  Temporal
    df["dow"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month

    # 4️⃣  Days‑since‑last
    df["home_prev_date"] = df.groupby("home_team")["date"].shift(1)
    df["away_prev_date"] = df.groupby("away_team")["date"].shift(1)
    df["home_days_since"] = (df["date"] - df["home_prev_date"]).dt.days
    df["away_days_since"] = (df["date"] - df["away_prev_date"]).dt.days

    # 5️⃣  Elo rating (online update)
    elo_dict: dict[str, float] = {}
    elo_home, elo_away = [], []
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        elo_dict.setdefault(h, 1500.0)
        elo_dict.setdefault(a, 1500.0)
        elo_home.append(elo_dict[h])
        elo_away.append(elo_dict[a])
        score_home = (
            1.0 if row["result"] == "H" else 0.5 if row["result"] == "D" else 0.0
        )
        exp_home = 1 / (1 + 10 ** ((elo_dict[a] - elo_dict[h]) / 400))
        elo_dict[h] += K_ELO * (score_home - exp_home)
        elo_dict[a] += K_ELO * ((1 - score_home) - (1 - exp_home))
    df["elo_home"], df["elo_away"] = elo_home, elo_away

    # 6️⃣  Poisson‑lambda (średnia goli do momentu meczu)
    grp_home = df.groupby("home_team")
    df["lambda_home_for"] = grp_home["home_goals"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df["lambda_home_against"] = grp_home["away_goals"].transform(
        lambda s: s.shift().expanding().mean()
    )

    grp_away = df.groupby("away_team")
    df["lambda_away_for"] = grp_away["away_goals"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df["lambda_away_against"] = grp_away["home_goals"].transform(
        lambda s: s.shift().expanding().mean()
    )

    # 7️⃣  Posprzątaj techniczne kolumny
    df.drop(columns=["home_prev_date", "away_prev_date"], inplace=True)

    return df
