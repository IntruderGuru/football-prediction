import warnings
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from joblib import load

from src.constants import DATA_PATH, MODEL_PATH, SCHEMA_PATH, FEATURE_COLUMNS
from src.features.build import extract_features


MIN_MATCHES = 5  # ile spotkań musi mieć drużyna, by rolling-statystyki miały sens


def _latest_team_row(fe_df: pd.DataFrame, team: str, side: str) -> pd.Series | None:
    """
    Zwraca ostatni wiersz z wyekstrahowanymi cechami dla drużyny
    (side = 'home' albo 'away').
    """
    mask = fe_df[f"{side}_team"] == team
    if mask.any():
        return fe_df[mask].iloc[-1]
    return None


def _league_means(fe_df: pd.DataFrame) -> pd.Series:
    """Średnie ligowe dla cech bukmacherskich (fallback)."""
    return fe_df[["bookie_prob_home", "bookie_prob_draw", "bookie_prob_away"]].mean()


def _warn_cold_start(team: str, n_matches: int):
    warnings.warn(
        f"⚠️  Drużyna '{team}' ma tylko {n_matches} meczów w historii – "
        "rolling statystyki mogą być mało wiarygodne."
    )


# scripts/infer_match.py (fragment)
def build_feature_vector(
    home_team: str,
    away_team: str,
    match_date: str | datetime,
    bookie_home: float | None = None,
    bookie_draw: float | None = None,
    bookie_away: float | None = None,
    update_history: bool = True,  # NEW
) -> pd.DataFrame:
    """Zwraca pojedynczy wiersz cech.
    • update_history=True  – jak dotychczas (używane przy predict-match)
    • update_history=False – NIE dopisuje fikcyjnego meczu (symulacja sezonu)
    """

    match_dt = pd.to_datetime(match_date)

    # 1. Historia do T-1
    hist_df = pd.read_parquet(DATA_PATH / "processed/model_input.parquet")
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    hist_df = hist_df[hist_df["date"] < match_dt].copy()

    # -------------- DOPISYWANIE PUSTEGO MECZU? -----------------
    if update_history:
        empty_row = {
            "date": match_dt,
            "home_team": home_team,
            "away_team": away_team,
            # reszta kolumn na None/NaN
        }
        for c in hist_df.columns:
            empty_row.setdefault(c, None)
        hist_df = pd.concat([hist_df, pd.DataFrame([empty_row])], ignore_index=True)
    # -----------------------------------------------------------

    fe_hist = extract_features(hist_df)
    ...

    # 3. Pobierz “aktualny stan” obu zespołów
    row_home = _latest_team_row(fe_hist, home_team, "home")
    row_away = _latest_team_row(fe_hist, away_team, "away")

    # — fallback: ostrzeż, jeżeli mało spotkań —
    n_home = len(
        hist_df[
            (hist_df["home_team"] == home_team) | (hist_df["away_team"] == home_team)
        ]
    )
    n_away = len(
        hist_df[
            (hist_df["home_team"] == away_team) | (hist_df["away_team"] == away_team)
        ]
    )

    if n_home < MIN_MATCHES:
        _warn_cold_start(home_team, n_home)
    if n_away < MIN_MATCHES:
        _warn_cold_start(away_team, n_away)

    # 4. Złożenie wektora
    vec = {}

    # xG shifty
    vec["xG_home_shift"] = (
        row_home["xG_home_shift"]
        if row_home is not None
        else fe_hist["xG_home_shift"].mean()
    )
    vec["xG_away_shift"] = (
        row_away["xG_away_shift"]
        if row_away is not None
        else fe_hist["xG_away_shift"].mean()
    )
    vec["xg_diff"] = vec["xG_home_shift"] - vec["xG_away_shift"]

    # Bukmacherka
    if None not in (bookie_home, bookie_draw, bookie_away):
        probs = pd.Series(
            [bookie_home, bookie_draw, bookie_away],
            index=["bookie_prob_home", "bookie_prob_draw", "bookie_prob_away"],
        )
    else:
        probs = _league_means(fe_hist)
    vec.update(probs.to_dict())

    # Rolling / forma – pobieramy z ostatnich wierszy; jak brak → średnia ligowa
    for col_home, col_away in [
        ("home_roll_xg_5", "away_roll_xg_5"),
        ("home_roll_gd_5", "away_roll_gd_5"),
        ("home_roll_form_5", "away_roll_form_5"),
    ]:
        vec[col_home] = (
            row_home[col_home] if row_home is not None else fe_hist[col_home].mean()
        )
        vec[col_away] = (
            row_away[col_away] if row_away is not None else fe_hist[col_away].mean()
        )

    # Data-time features
    vec["dow"] = match_dt.weekday()
    vec["month"] = match_dt.month

    # days_since last match
    vec["home_days_since"] = (
        (match_dt - row_home["date"]).days if row_home is not None else 7
    )
    vec["away_days_since"] = (
        (match_dt - row_away["date"]).days if row_away is not None else 7
    )

    # Elo
    vec["elo_home"] = row_home["elo_home"] if row_home is not None else 1500
    vec["elo_away"] = row_away["elo_away"] if row_away is not None else 1500
    vec["elo_diff"] = vec["elo_home"] - vec["elo_away"]

    # Lambdy Poissona
    for col in [
        "lambda_home_for",
        "lambda_home_against",
        "lambda_away_for",
        "lambda_away_against",
    ]:
        if col.startswith("lambda_home"):
            vec[col] = row_home[col] if row_home is not None else fe_hist[col].mean()
        else:
            vec[col] = row_away[col] if row_away is not None else fe_hist[col].mean()

    # 5. DataFrame w kolejności FEATURE_COLUMNS
    return pd.DataFrame([[vec[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


def predict_match(home_team: str, away_team: str, date: str) -> dict:
    """Zwraca predykcję i p-y dla pojedynczego meczu."""
    # — budujemy wektor cech —
    X = build_feature_vector(home_team, away_team, date)

    # — model + schema —
    model = load(MODEL_PATH)
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    # (upewnij się, że schema == FEATURE_COLUMNS)
    X = X[schema]

    # — predykcja —
    prob = model.predict_proba(X)[0]
    pred = model.classes_[prob.argmax()]

    return {
        "prediction": pred,
        "probabilities": {cls: float(p) for cls, p in zip(model.classes_, prob)},
    }
