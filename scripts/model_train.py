import json
import argparse
import joblib
import os
from src.data_loader import load_data
from src.features import extract_features
from src.model import train_model

NUMERIC_FEATURES = [
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="rf", choices=["rf", "lgb"])
    parser.add_argument(
        "--params", default=None, help="path to JSON with tuned hyper-params"
    )
    args = parser.parse_args()

    df = extract_features(load_data()).dropna(subset=NUMERIC_FEATURES + ["result"])
    X, y = df[NUMERIC_FEATURES], df["result"]

    params = {}
    if args.params:
        with open(args.params) as f:
            params = json.load(f)

    model = train_model(X, y, algo=args.algo, params=params)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/final_model_{args.algo}.pkl")
    print(f"Model saved to models/final_model_{args.algo}.pkl")


if __name__ == "__main__":
    main()
