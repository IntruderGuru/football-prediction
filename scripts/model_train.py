import json
import argparse
import joblib
import os
from src.data_loader import load_data
from src.features import extract_features
from src.model import train_model
from src.constants import FEATURE_COLUMNS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="rf", choices=["rf", "lgb", "cat", "xgb"])
    parser.add_argument(
        "--params", default=None, help="path to JSON with tuned hyper-params"
    )
    args = parser.parse_args()

    df = extract_features(load_data()).dropna(subset=FEATURE_COLUMNS + ["result"])
    X, y = df[FEATURE_COLUMNS], df["result"]

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
