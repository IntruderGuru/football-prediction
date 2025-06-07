import json
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

FEATURE_ORDER = [
    "xG_home",
    "xG_away",
    "bookie_prob_home",
    "bookie_prob_draw",
    "bookie_prob_away",
    "home_roll_xg_5",
    "away_roll_xg_5",
    "home_roll_gd_5",
    "away_roll_gd_5",
    "home_roll_form_5",
    "away_roll_form_5",
    "dow",
    "month",
    "home_days_since",
    "away_days_since",
]


def load_features(args):
    if args.features_file:
        feats = json.loads(Path(args.features_file).read_text())
    else:
        feats = json.loads(args.features)
    missing = [c for c in FEATURE_ORDER if c not in feats]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return pd.DataFrame([[feats[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="lgb", choices=["rf", "lgb"])
    ap.add_argument("--model_path", default="models/final_model_lgb.pkl")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--features", help="inline JSON string with features")
    g.add_argument("--features_file", help="path to JSON file with features")
    args = ap.parse_args()

    model = joblib.load(args.model_path)
    X_new = load_features(args)
    proba = model.predict_proba(X_new)[0]
    pred = model.classes_[np.argmax(proba)]

    print(f"Prediction: {pred}")
    print(f"Probabilities (H/D/A): {dict(zip(model.classes_, proba.round(3)))}")


if __name__ == "__main__":
    main()
