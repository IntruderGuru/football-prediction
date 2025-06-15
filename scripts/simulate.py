import pandas as pd


def simulate_match_input(
    df: pd.DataFrame, home_team: str, away_team: str, match_date: str
) -> pd.DataFrame:
    """
    Build a feature row for a specific match date:
      - Past matches: state before the match
      - Future matches: most recent state
    """
    date = pd.to_datetime(match_date)

    # filter team matches
    home_df = df[(df["home_team"] == home_team) | (df["away_team"] == home_team)]
    away_df = df[(df["home_team"] == away_team) | (df["away_team"] == away_team)]

    # restrict to before match date
    home_df = home_df[home_df["date"] < date]
    away_df = away_df[away_df["date"] < date]

    if home_df.empty or away_df.empty:
        raise ValueError("Not enough past data to simulate features.")

    # select latest state
    home_row = home_df.sort_values("date").iloc[-1]
    away_row = away_df.sort_values("date").iloc[-1]

    # combine home and away features
    row = home_row.copy()
    for col in away_row.index:
        if col.startswith("away_"):
            row[col] = away_row[col]
    row["home_team"] = home_team
    row["away_team"] = away_team
    row["date"] = date

    return row.to_frame().T  # single-row DataFrame
