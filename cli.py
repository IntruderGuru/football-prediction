import argparse
import joblib
import pandas as pd

from src.pipeline import run_pipeline
from src.model import evaluate_model
from src.data_loader import load_data
from src.features import extract_features
from src.constants import FEATURE_COLUMNS


def main():
    parser = argparse.ArgumentParser(description="Football Prediction CLI")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--algo", choices=["rf", "lgb", "cat", "stack"], default="lgb"
    )
    train_parser.add_argument("--input", default="data/processed/model_input.parquet")
    train_parser.add_argument("--output", default="output")
    train_parser.add_argument("--save", action="store_true")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate saved model")
    eval_parser.add_argument("--model-path", required=True)
    eval_parser.add_argument("--input", default="data/processed/model_input.parquet")

    # Predict
    predict_parser = subparsers.add_parser("predict", help="Predict a single match")
    predict_parser.add_argument("--home", required=True)
    predict_parser.add_argument("--away", required=True)
    predict_parser.add_argument("--date", required=True)
    predict_parser.add_argument("--model-path", required=True)

    args = parser.parse_args()

    if args.command == "train":
        run_pipeline(
            algo=args.algo,
            input_path=args.input,
            output_dir=args.output,
            save=args.save,
        )

    elif args.command == "evaluate":
        model = joblib.load(args.model_path)
        df = extract_features(load_data(args.input)).dropna(subset=FEATURE_COLUMNS)
        X, y = df[FEATURE_COLUMNS], df["result"]
        y_pred = model.predict(X)
        metrics = evaluate_model(y, y_pred, verbose=True)
        print(
            f"Evaluation - Accuracy: {metrics['accuracy']:.3f} | Macro-F1: {metrics['macro_f1']:.3f}"
        )

    elif args.command == "predict":
        model = joblib.load(args.model_path)
        df = extract_features(load_data("data/processed/model_input.parquet")).dropna(
            subset=FEATURE_COLUMNS
        )
        match_row = df.query(
            "home_team == @args.home and away_team == @args.away and date == @args.date"
        )
        if match_row.empty:
            print("❌ Match not found.")
            return
        X = match_row[FEATURE_COLUMNS]
        pred = model.predict(X)[0]
        print(f"⚽ Prediction for {args.home} vs {args.away} on {args.date}: {pred}")


if __name__ == "__main__":
    main()
