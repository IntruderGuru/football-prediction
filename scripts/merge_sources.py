import pandas as pd
import ast
import numpy as np
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from src.constants import TEAM_MAP


def merge_sources():
    """
    Merge raw CSVs and Understat data into processed Parquet files:
      - Parse and normalize dates
      - Extract xG and goals from JSON fields
      - Normalize team and league names
      - Fill missing xG with seasonal averages
      - Compute bookmaker probabilities and result labels
    """
    # load merged football-data.co.uk
    f = pd.read_csv("data/raw/football_data_merged.csv", parse_dates=["date"])
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    f["date_only"] = f["date"].dt.normalize()

    # load Understat JSON data
    u = pd.read_csv("data/raw/understat_all.csv")
    u["datetime"] = pd.to_datetime(u["datetime"], errors="coerce")
    u = u.dropna(subset=["datetime"])
    u["date"] = u["datetime"].dt.normalize()

    # helper to parse JSON fields
    def parse_json(val, key, cast):
        try:
            return cast(ast.literal_eval(val)[key])
        except Exception:
            return np.nan

    # extract xG and goals
    u["xG_home"] = u["xG"].apply(lambda s: parse_json(s, "h", float))
    u["xG_away"] = u["xG"].apply(lambda s: parse_json(s, "a", float))
    u["home_goals_us"] = u["goals"].apply(lambda s: parse_json(s, "h", int))
    u["away_goals_us"] = u["goals"].apply(lambda s: parse_json(s, "a", int))

    # normalize team names
    u["home_team_normed"] = u["h"].apply(
        lambda x: TEAM_MAP.get(
            ast.literal_eval(x)["title"].strip(), ast.literal_eval(x)["title"].strip()
        )
    )
    u["away_team_normed"] = u["a"].apply(
        lambda x: TEAM_MAP.get(
            ast.literal_eval(x)["title"].strip(), ast.literal_eval(x)["title"].strip()
        )
    )

    # upper-case league codes
    f["league_normed"] = f["league"].str.strip().str.upper()
    u["league_normed"] = u["league"].str.strip().str.upper()

    # select columns for merge
    u_sub = u[
        [
            "date",
            "league_normed",
            "home_team_normed",
            "away_team_normed",
            "xG_home",
            "xG_away",
        ]
    ]
    f_sub = f[
        [
            "date",
            "league",
            "league_normed",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "bookie_home",
            "bookie_draw",
            "bookie_away",
        ]
    ]

    # merge sources
    merged = f_sub.merge(
        u_sub,
        left_on=["date", "league_normed", "home_team", "away_team"],
        right_on=["date", "league_normed", "home_team_normed", "away_team_normed"],
        how="left",
    )

    # fill missing xG with seasonal averages
    merged["season"] = merged["date"].dt.year
    avg_xg = (
        merged.groupby(["league_normed", "season"])
        .agg(avg_xg_home=("xG_home", "mean"), avg_xg_away=("xG_away", "mean"))
        .reset_index()
    )
    merged = merged.merge(avg_xg, on=["league_normed", "season"], how="left")
    merged["xG_home"].fillna(merged["avg_xg_home"], inplace=True)
    merged["xG_away"].fillna(merged["avg_xg_away"], inplace=True)

    # compute bookmaker probabilities
    odds = 1 / merged[["bookie_home", "bookie_draw", "bookie_away"]]
    sum_inv = odds.sum(axis=1)
    merged["bookie_prob_home"] = odds["bookie_home"] / sum_inv
    merged["bookie_prob_draw"] = odds["bookie_draw"] / sum_inv
    merged["bookie_prob_away"] = odds["bookie_away"] / sum_inv

    # label match result
    merged["result"] = np.select(
        [
            merged["home_goals"] > merged["away_goals"],
            merged["home_goals"] < merged["away_goals"],
        ],
        ["H", "A"],
        default="D",
    )

    # keep only needed columns
    cols = [
        "date",
        "league",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "bookie_home",
        "bookie_draw",
        "bookie_away",
        "xG_home",
        "xG_away",
        "bookie_prob_home",
        "bookie_prob_draw",
        "bookie_prob_away",
        "result",
    ]
    processed = merged[cols]

    # write outputs
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    pa_table = pa.Table.from_pandas(merged, preserve_index=False)
    pq.write_table(pa_table, "data/processed/merged.parquet")

    model_input = processed.dropna(
        subset=["xG_home", "xG_away", "bookie_home", "bookie_draw", "bookie_away"]
    )
    schema = pa.schema(
        [
            ("date", pa.timestamp("ns")),
            ("home_team", pa.string()),
            ("away_team", pa.string()),
            ("home_goals", pa.int64()),
            ("away_goals", pa.int64()),
            ("bookie_home", pa.float64()),
            ("bookie_draw", pa.float64()),
            ("bookie_away", pa.float64()),
            ("league", pa.string()),
            ("xG_home", pa.float64()),
            ("xG_away", pa.float64()),
            ("bookie_prob_home", pa.float64()),
            ("bookie_prob_draw", pa.float64()),
            ("bookie_prob_away", pa.float64()),
            ("result", pa.string()),
        ]
    )
    pq.write_table(
        pa.Table.from_pandas(model_input, schema=schema, preserve_index=False),
        "data/processed/model_input.parquet",
    )

    # export text summary
    model_input.to_string(index=False)


if __name__ == "__main__":
    merge_sources()
