import json
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
from joblib import load

from src.constants import DATA_PATH, MODEL_PATH, SCHEMA_PATH
from scripts.infer_match import build_feature_vector

# ─────────────────────────── CONFIG ──────────────────────────── #
POINTS = {"H": (3, 0), "A": (0, 3), "D": (1, 1)}
ROUND_GAP_DAYS = 7  # odstęp pomiędzy kolejkami (1 tydzień)
TEAM_COUNT = 20  # stała liczba zespołów w top‑ligach
MIN_MATCHES = 10  # ile meczów drużyna musi mieć w sezonie, żeby ją uwzględnić

# ────────────────────────── HELPERS ──────────────────────────── #


def _season_boundaries(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Sezon = 1 lipca <year> → 30 czerwca <year+1>."""
    start = pd.Timestamp(year=year, month=7, day=1)
    end = pd.Timestamp(year=year + 1, month=6, day=30)
    return start, end


def _teams_for_season(league: str, year: int, hist: pd.DataFrame) -> List[str]:
    """Zwraca dokładnie TEAM_COUNT drużyn z danej ligi w konkretnym sezonie."""
    start, end = _season_boundaries(year)
    mask = (
        (hist["league"].str.upper() == league.upper())
        & (hist["date"] >= start)
        & (hist["date"] <= end)
    )
    season_hist = hist.loc[mask]
    if season_hist.empty:
        raise ValueError("Brak danych meczowych dla podanego sezonu.")

    counts = pd.concat(
        [season_hist["home_team"], season_hist["away_team"]]
    ).value_counts()
    legit = counts[counts >= MIN_MATCHES].head(TEAM_COUNT)
    if len(legit) < TEAM_COUNT:
        raise ValueError(
            f"Znaleziono {len(legit)} drużyn z ≥{MIN_MATCHES} meczami. Oczekiwano {TEAM_COUNT}."
        )
    return legit.index.tolist()


def _circle_round_robin(teams: List[str]) -> List[List[tuple]]:
    """Generuje terminarz (algorytm circle‑method)."""
    n = len(teams)
    half = n // 2
    rot = teams[1:]
    rounds = []
    for _ in range(n - 1):
        left = [teams[0]] + rot[: half - 1]
        right = rot[half - 1 :][::-1]
        rounds.append(list(zip(left, right)))
        # rotacja
        rot = rot[1:] + rot[:1]
    return rounds


def _fixtures(teams: List[str], first_round_date: pd.Timestamp) -> pd.DataFrame:
    """Double round‑robin: 38 kolejek dla 20 drużyn."""
    single = _circle_round_robin(teams)
    double = single + [[(b, a) for a, b in rnd] for rnd in single]  # rewanże

    rows = []
    for rnd_idx, matches in enumerate(double, start=1):
        dt = first_round_date + timedelta(days=(rnd_idx - 1) * ROUND_GAP_DAYS)
        for home, away in matches:
            rows.append(
                {"round": rnd_idx, "date": dt, "home_team": home, "away_team": away}
            )
    return pd.DataFrame(rows)


def _init_table(teams: List[str]) -> Dict[str, Dict[str, int]]:
    return {t: {"PTS": 0, "W": 0, "D": 0, "L": 0} for t in teams}


def _update_table(tbl: Dict[str, Dict[str, int]], home: str, away: str, res: str):
    ph, pa = POINTS[res]
    th, ta = tbl[home], tbl[away]
    th["PTS"] += ph
    ta["PTS"] += pa
    if res == "H":
        th["W"] += 1
        ta["L"] += 1
    elif res == "A":
        ta["W"] += 1
        th["L"] += 1
    else:
        th["D"] += 1
        ta["D"] += 1


def _table_df(tbl: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    df = (
        pd.DataFrame.from_dict(tbl, orient="index")
        .sort_values(["PTS", "W"], ascending=False)
        .reset_index()
        .rename(columns={"index": "Team"})
    )
    df.index += 1
    return df


# ────────────────────────── CORE ──────────────────────────── #


def simulate_league(
    league: str,
    season_year: int,
    model_path: Path = MODEL_PATH,
    schema_path: Path = SCHEMA_PATH,
    first_round_day_month: tuple[int, int] = (15, 8),  # 15 sierpnia domyślnie
    show_rounds: bool = False,
):
    start_hist, _ = _season_boundaries(season_year)

    hist = pd.read_parquet(DATA_PATH / "processed/model_input.parquet")
    teams = _teams_for_season(league, season_year, hist)

    first_round_date = pd.Timestamp(
        year=season_year, month=first_round_day_month[1], day=first_round_day_month[0]
    )
    schedule = _fixtures(teams, first_round_date)

    model = load(model_path)
    with open(schema_path) as f:
        schema = json.load(f)

    table = _init_table(teams)
    curr_round = 0
    for fx in schedule.itertuples(index=False):
        if show_rounds and fx.round != curr_round:
            if curr_round:
                print(f"\n=== Kolejka {curr_round} ===")
                print(_table_df(table).to_string())
            curr_round = fx.round

        X = build_feature_vector(
            fx.home_team, fx.away_team, fx.date, update_history=False  # ← poprawka
        )[schema]
        res = model.predict(X)[0]
        _update_table(table, fx.home_team, fx.away_team, res)

    print("\n=== TABELA KOŃCOWA ===")
    print(_table_df(table).to_string())


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser("Symulacja sezonu ligowego")
    cli.add_argument("league", help="Kod ligi, np. LALIGA")
    cli.add_argument("season", type=int, help="Rok rozpoczęcia sezonu, np. 2023")
    cli.add_argument("--rounds", action="store_true", help="Pokaż tabelę co kolejkę")
    cli.add_argument(
        "--day", type=int, default=15, help="Dzień 1. kolejki (default 15)"
    )
    cli.add_argument(
        "--month", type=int, default=8, help="Miesiąc 1. kolejki (default 8)"
    )
    args = cli.parse_args()

    simulate_league(
        args.league,
        args.season,
        first_round_day_month=(args.day, args.month),
        show_rounds=args.rounds,
    )
