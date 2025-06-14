import argparse, json, joblib
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import poisson

from src.features import extract_features
from src.constants import FEATURE_COLUMNS

MODEL_PATH = Path("output/final_model_lgb.pkl")
SCHEMA_PATH = Path("output/feature_schema.json")
DATA_PATH = Path("data/processed/model_input.parquet")


def poisson_probs(lam_hf, lam_ha, lam_af, lam_aa, max_goals=8):

    p_home = p_draw = p_away = 0.0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = poisson.pmf(hg, lam_hf) * poisson.pmf(ag, lam_af)
            if hg > ag:
                p_home += p
            elif hg < ag:
                p_away += p
            else:
                p_draw += p
    s = p_home + p_draw + p_away
    return p_home / s, p_draw / s, p_away / s


def build_feature_row(home, away, date, odds):

    df_hist = pd.read_parquet(DATA_PATH)
    df_hist = df_hist[df_hist["date"] < date]

    fake = {
        "date": date,
        "home_team": home,
        "away_team": away,
        "home_goals": np.nan,
        "away_goals": np.nan,
        "xG_home": np.nan,
        "xG_away": np.nan,
        "bookie_prob_home": np.nan,
        "bookie_prob_draw": np.nan,
        "bookie_prob_away": np.nan,
    }
    df = pd.concat([df_hist, pd.DataFrame([fake])], ignore_index=True)
    features = extract_features(df).iloc[-1]

    if odds is None:
        p_home, p_draw, p_away = poisson_probs(
            features["lambda_home_for"],
            features["lambda_home_against"],
            features["lambda_away_for"],
            features["lambda_away_against"],
        )
    else:
        p_home, p_draw, p_away = odds

    features[["bookie_prob_home", "bookie_prob_draw", "bookie_prob_away"]] = [
        p_home,
        p_draw,
        p_away,
    ]
    row = features[FEATURE_COLUMNS].astype("float64")
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--odds-home", type=float)
    ap.add_argument("--odds-draw", type=float)
    ap.add_argument("--odds-away", type=float)
    args = ap.parse_args()

    date_obj = pd.to_datetime(args.date)
    odds = None
    if args.odds_home and args.odds_draw and args.odds_away:
        inv = [1 / x for x in (args.odds_home, args.odds_draw, args.odds_away)]
        s = sum(inv)
        odds = [x / s for x in inv]

    X = build_feature_row(args.home, args.away, date_obj, odds)
    model = joblib.load(MODEL_PATH)

    proba = model.predict_proba(X)[0]
    pred = model.classes_[np.argmax(proba)]

    print(f"{args.home} - {args.away}  ({args.date})")
    print(
        f"P(H)={proba[model.classes_=='H'][0]:.2%} · "
        f"P(D)={proba[model.classes_=='D'][0]:.2%} · "
        f"P(A)={proba[model.classes_=='A'][0]:.2%}"
    )
    print(f"Prediction: {pred}")


if __name__ == "__main__":
    main()
