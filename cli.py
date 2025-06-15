import argparse
import joblib
import pandas as pd

from src.pipeline import run_pipeline, get_evaluation_split
from src.model import evaluate_model
from src.data_loader import load_data
from src.features import extract_features
from scripts.simulate import simulate_match_input
from src.constants import FEATURE_COLUMNS

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Football Prediction CLI")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--algo", choices=["rf", "lgb", "cat", "stack"], default="lgb"
    )
    train_parser.add_argument("--input", default="data/processed/model_input.parquet")
    train_parser.add_argument("--output", default="models")
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
    predict_parser.add_argument(
        "--verbose", action="store_true", help="Show feature values used for prediction"
    )
    predict_parser.add_argument(
        "--proba", action="store_true", help="Show predicted class probabilities"
    )

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
        X, y = get_evaluation_split(args.input)
        y_pred = model.predict(X)
        metrics = evaluate_model(y, y_pred, X_eval=X, model=model, verbose=True)
        print(
            f"Evaluation - Accuracy: {metrics['accuracy']:.3f} | Macro-F1: {metrics['macro_f1']:.3f} | log_loss: {metrics['log_loss']:.4f}"
        )
        importances = model.feature_importances_
        features = X.columns
        sorted_idx = importances.argsort()[::-1]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), importances[sorted_idx])
        plt.yticks(range(len(features)), [features[i] for i in sorted_idx])
        plt.title("Feature Importances")
        plt.gca().invert_yaxis()
        plt.show()

    elif args.command == "predict":
        model = joblib.load(args.model_path)
        df = extract_features(load_data("data/processed/model_input.parquet")).dropna(
            subset=FEATURE_COLUMNS
        )
        match_row = simulate_match_input(df, args.home, args.away, args.date)
        if match_row.empty:
            print("❌ Match not found.")
            return
        X = match_row[FEATURE_COLUMNS].astype(float)

        pred = model.predict(X).item()
        print(f"\n⚽ Prediction for {args.home} vs {args.away} on {args.date}: {pred}")

        if args.proba:
            try:
                proba = model.predict_proba(X)[0]
                classes = (
                    model.classes_ if hasattr(model, "classes_") else ["H", "D", "A"]
                )
                proba_str = ", ".join(
                    f"{cls}: {100*p:.1f}%" for cls, p in zip(classes, proba)
                )
                print(f"🔎 Probabilities: {proba_str}")
            except Exception as e:
                print(f"⚠️ Could not retrieve probabilities: {e}")

        if args.verbose:
            print("\n🧾 Feature values:")
            for col in FEATURE_COLUMNS:
                val = match_row.iloc[0][col]
                print(f"   {col:>20}: {val}")


if __name__ == "__main__":
    main()
