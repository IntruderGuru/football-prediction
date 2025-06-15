import pandas as pd
from datetime import datetime


def simulate_match_input(
    df: pd.DataFrame, home_team: str, away_team: str, match_date: str
) -> pd.DataFrame:
    """
    Given the full dataset, simulate the feature row for a future or past match.
    If the match is in the past, extract features as they were before the match.
    If the match is in the future, use the latest available data.
    """
    match_date_dt = pd.to_datetime(match_date)

    # Filter past matches for each team
    home_past_matches = df[
        (df["home_team"] == home_team) | (df["away_team"] == home_team)
    ]
    away_past_matches = df[
        (df["home_team"] == away_team) | (df["away_team"] == away_team)
    ]

    # Restrict to only matches BEFORE the given date
    home_past_matches = home_past_matches[home_past_matches["date"] < match_date_dt]
    away_past_matches = away_past_matches[away_past_matches["date"] < match_date_dt]

    if home_past_matches.empty or away_past_matches.empty:
        raise ValueError("❌ Not enough past data for the teams to generate features.")

    # Take the latest state of each team before the match
    latest_home_row = home_past_matches.sort_values("date").iloc[-1]
    latest_away_row = away_past_matches.sort_values("date").iloc[-1]

    # Generate a synthetic row using latest state
    example_row = latest_home_row.copy()
    for col in latest_away_row.index:
        if col.startswith("away_"):
            example_row[col] = latest_away_row[col]
    example_row["home_team"] = home_team
    example_row["away_team"] = away_team
    example_row["date"] = match_date_dt

    return example_row.to_frame().T  # return as DataFrame with 1 row
