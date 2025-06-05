import pandas as pd
import ast
import numpy as np
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


def merge_sources():

    f = pd.read_csv("data/raw/football_data_merged.csv", parse_dates=["date"])

    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    f["date_only"] = f["date"].dt.normalize()

    u = pd.read_csv("data/raw/understat_all.csv")
    u["datetime"] = pd.to_datetime(u["datetime"], errors="coerce")
    u = u[u["datetime"].notna()]
    u["date"] = u["datetime"].dt.normalize()

    def parse_json(x, key, cast_type):
        try:
            return cast_type(ast.literal_eval(x)[key])
        except (ValueError, SyntaxError, KeyError, TypeError):
            return np.nan

    u["home_goals_us"] = u["goals"].apply(lambda s: parse_json(s, "h", int))
    u["away_goals_us"] = u["goals"].apply(lambda s: parse_json(s, "a", int))
    u["xG_home"] = u["xG"].apply(lambda s: parse_json(s, "h", float))
    u["xG_away"] = u["xG"].apply(lambda s: parse_json(s, "a", float))

    team_map = {
        # ——— Premier League / Championship
        "Manchester United": "Man United",
        "Manchester City": "Man City",
        "Wolverhampton Wanderers": "Wolves",
        "Brighton & Hove Albion": "Brighton",
        "Tottenham Hotspur": "Tottenham",
        "West Ham United": "West Ham",
        "Newcastle United": "Newcastle",
        "Nottingham Forest": "Nott'm Forest",
        "Sheffield United": "Sheffield United",
        "Crystal Palace": "Crystal Palace",
        "Liverpool": "Liverpool",
        "Chelsea": "Chelsea",
        "Arsenal": "Arsenal",
        "Everton": "Everton",
        "Bournemouth": "Bournemouth",
        "Burnley": "Burnley",
        "Southampton": "Southampton",
        "Leicester": "Leicester",
        "Leeds": "Leeds",
        "Aston Villa": "Aston Villa",
        "Watford": "Watford",
        "Wolves": "Wolves",
        "West Bromwich Albion": "West Brom",
        "Fulham": "Fulham",
        # ——— La Liga
        "Athletic Club": "Ath Bilbao",
        "Atletico Madrid": "Ath Madrid",
        "Real Madrid": "Real Madrid",
        "FC Barcelona": "Barcelona",
        "Real Sociedad": "Sociedad",
        "Real Betis": "Betis",
        "Real Valladolid": "Valladolid",
        "Espanyol": "Espanol",
        "Granada": "Granada",
        "Celta Vigo": "Celta",
        "Sevilla": "Sevilla",
        "Valencia": "Valencia",
        "Villarreal": "Villarreal",
        "Getafe": "Getafe",
        "Levante": "Levante",
        "Alaves": "Alaves",
        "Mallorca": "Mallorca",
        "Elche": "Elche",
        "Cadiz": "Cadiz",
        "Eibar": "Eibar",
        "Osasuna": "Osasuna",
        # ——— Ligue 1
        "Paris Saint Germain": "Paris SG",
        "Olympique Marseille": "Marseille",
        "Lyon": "Lyon",
        "Lille": "Lille",
        "AS Monaco": "Monaco",
        "Nice": "Nice",
        "Stade Rennais": "Rennes",
        "Nantes": "Nantes",
        "Strasbourg": "Strasbourg",
        "Brest": "Brest",
        "Ajaccio": "Ajaccio",
        "Nimes": "Nimes",
        "Montpellier": "Montpellier",
        "Lorient": "Lorient",
        "Bordeaux": "Bordeaux",
        "Reims": "Reims",
        "Amiens": "Amiens",
        "Angers": "Angers",
        "Dijon": "Dijon",
        "Clermont Foot": "Clermont",
        "Metz": "Metz",
        "Lens": "Lens",
        # ——— Serie A
        "Inter Milan": "Inter",
        "AC Milan": "Milan",
        "Juventus": "Juventus",
        "Atalanta": "Atalanta",
        "Napoli": "Napoli",
        "Roma": "Roma",
        "SS Lazio": "Lazio",
        "Fiorentina": "Fiorentina",
        "Torino": "Torino",
        "Udinese": "Udinese",
        "Bologna": "Bologna",
        "Empoli": "Empoli",
        "Sassuolo": "Sassuolo",
        "Cagliari": "Cagliari",
        "Sampdoria": "Sampdoria",
        "Verona": "Verona",
        "Genoa": "Genoa",
        "Salernitana": "Salernitana",
        "Brescia": "Brescia",
        "Lecce": "Lecce",
        "Monza": "Monza",
        # ——— Bundesliga
        "Bayern Munich": "Bayern Munich",
        "Borussia Dortmund": "Dortmund",
        "Bayer Leverkusen": "Leverkusen",
        "RB Leipzig": "RB Leipzig",
        "Borussia M.Gladbach": "M'gladbach",
        "VfB Stuttgart": "Stuttgart",
        "Eintracht Frankfurt": "Ein Frankfurt",
        "FC Cologne": "FC Koln",
        "Union Berlin": "Union Berlin",
        "Freiburg": "Freiburg",
        "Hertha Berlin": "Hertha",
        "Hoffenheim": "Hoffenheim",
        "Wolfsburg": "Wolfsburg",
        "Mainz 05": "Mainz",
        "Augsburg": "Augsburg",
        "Schalke 04": "Schalke 04",
        "Werder Bremen": "Werder Bremen",
        "Fortuna Duesseldorf": "Fortuna Dusseldorf",
        "Greuther Fuerth": "Greuther Furth",
        "FC Heidenheim": "Heidenheim",
        "Arminia Bielefeld": "Bielefeld",
        # ——— Inne
        "Holstein Kiel": "Holstein Kiel",
        "Ipswich": "Ipswich",
        "Como": "Como",
        "Parma Calcio 1913": "Parma",
        "RasenBallsport Leipzig": "RB Leipzig",
        "Rayo Vallecano": "Vallecano",
        "Saint-Etienne": "St Etienne",
        "St. Pauli": "St. Pauli",
    }

    u["home_team_normed"] = u["h"].apply(
        lambda x: team_map.get(
            ast.literal_eval(x)["title"].strip(), ast.literal_eval(x)["title"].strip()
        )
    )
    u["away_team_normed"] = u["a"].apply(
        lambda x: team_map.get(
            ast.literal_eval(x)["title"].strip(), ast.literal_eval(x)["title"].strip()
        )
    )

    f["league_normed"] = f["league"].astype(str).str.strip().str.upper()
    u["league_normed"] = u["league"].astype(str).str.strip().str.upper()

    u_sub = u[
        [
            "date",
            "league_normed",
            "home_team_normed",
            "away_team_normed",
            "xG_home",
            "xG_away",
        ]
    ].copy()

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
    ].copy()

    merged = f_sub.merge(
        u_sub,
        left_on=["date", "league_normed", "home_team", "away_team"],
        right_on=["date", "league_normed", "home_team_normed", "away_team_normed"],
        how="left",
    )

    merged["season"] = merged["date"].dt.year

    avg_xg_home = (
        merged[merged["xG_home"].notna()]
        .groupby(["league_normed", "season"])["xG_home"]
        .mean()
        .reset_index()
        .rename(columns={"xG_home": "avg_xg_home"})
    )

    avg_xg_away = (
        merged[merged["xG_away"].notna()]
        .groupby(["league_normed", "season"])["xG_away"]
        .mean()
        .reset_index()
        .rename(columns={"xG_away": "avg_xg_away"})
    )

    merged = merged.merge(avg_xg_home, on=["league_normed", "season"], how="left")
    merged = merged.merge(avg_xg_away, on=["league_normed", "season"], how="left")

    mask_home = merged["xG_home"].isna()
    merged.loc[mask_home, "xG_home"] = merged.loc[mask_home, "avg_xg_home"]

    mask_away = merged["xG_away"].isna()
    merged.loc[mask_away, "xG_away"] = merged.loc[mask_away, "avg_xg_away"]

    merged["bookie_sum"] = (
        1.0 / merged["bookie_home"]
        + 1.0 / merged["bookie_draw"]
        + 1.0 / merged["bookie_away"]
    )
    merged["bookie_prob_home"] = (1.0 / merged["bookie_home"]) / merged["bookie_sum"]
    merged["bookie_prob_draw"] = (1.0 / merged["bookie_draw"]) / merged["bookie_sum"]
    merged["bookie_prob_away"] = (1.0 / merged["bookie_away"]) / merged["bookie_sum"]

    merged["result"] = np.select(
        [
            merged["home_goals"] > merged["away_goals"],
            merged["home_goals"] < merged["away_goals"],
        ],
        ["H", "A"],
        default="D",
    )

    merged = merged[
        [
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
            "bookie_sum",
            "bookie_prob_home",
            "bookie_prob_draw",
            "bookie_prob_away",
            "result",
        ]
    ].copy()

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    table_all = pa.Table.from_pandas(merged, preserve_index=False)
    pq.write_table(table_all, "data/processed/merged.parquet")

    model_input = merged.dropna(
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
            ("bookie_sum", pa.float64()),
            ("bookie_prob_home", pa.float64()),
            ("bookie_prob_draw", pa.float64()),
            ("bookie_prob_away", pa.float64()),
            ("result", pa.string()),
        ]
    )

    table_model = pa.Table.from_pandas(model_input, schema=schema, preserve_index=False)
    pq.write_table(table_model, "data/processed/model_input.parquet")

    with open("data/processed/model_input.txt", "w", encoding="utf-8") as fout:
        fout.write(model_input.to_string(index=False))


if __name__ == "__main__":
    merge_sources()
