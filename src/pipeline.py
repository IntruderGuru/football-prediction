import argparse
import json
import logging
from pathlib import Path

import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE

from src.constants import FEATURE_COLUMNS
from src.data_loader import load_data
from src.features import extract_features
from src.model import train_model, evaluate_model, get_model
from src.metrics import (
    save_classification_report_txt,
    save_classification_report_json,
    save_confusion_matrix_plot,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("footpred")


def run_pipeline(
    *,
    algo: str = "lgb",  # model to train: "rf", "cat", "lgb", or "stack"
    input_path: str = "data/processed/model_input.parquet",
    output_dir: str = "models",
    save: bool = False,  # whether to save model and artifacts
) -> None:
    logger.info("Loading data from → %s", input_path)
    df = extract_features(load_data(input_path)).dropna(subset=FEATURE_COLUMNS)
    X, y = df[FEATURE_COLUMNS], df["result"]

    print("Class distribution in target variable:")
    print(y.value_counts())

    # TimeSeriesSplit - last fold will serve as holdout test set
    cv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(cv.split(X))[-1]
    X_tr, X_te, y_tr, y_te = (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    )

    logger.info(
        "Training range: %s → %s",
        df.iloc[train_idx]["date"].min(),
        df.iloc[train_idx]["date"].max(),
    )
    logger.info(
        "Test range:     %s → %s",
        df.iloc[test_idx]["date"].min(),
        df.iloc[test_idx]["date"].max(),
    )

    # Handle class imbalance using SMOTE
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_tr, y_tr)
    print("After SMOTE class distribution:", y_resampled.value_counts())
    print("y_resampled type:", type(y_resampled.iloc[0]))

    # Train the selected model
    if algo == "stack":
        base = [
            ("lgb", get_model(params={"class_weight": None})),
            ("cat", get_model("cat")),
            ("rf", get_model(params={"class_weight": None})),
        ]
        model = StackingClassifier(
            estimators=base,
            final_estimator=get_model("lgb", params={"class_weight": None}),
            passthrough=True,
            n_jobs=-1,
        ).fit(X_resampled, y_resampled)
    else:
        params = {}
        if algo == "lgb":
            tuning_path = Path("output/lgb_best.json")
            if tuning_path.exists():
                logger.info("Loading best LGBM parameters from tuning")
                params = json.loads(tuning_path.read_text())
            else:
                logger.warning(
                    "File output/lgb_best.json not found - using default LGBM parameters"
                )

        model = train_model(X_resampled, y_resampled, algo=algo, params=params)

    # Evaluate on the time-based holdout set
    y_pred = model.predict(X_te)
    metrics = evaluate_model(y_te, y_pred, verbose=False)
    logger.info(
        "Evaluation - Accuracy: %.3f | Macro-F1: %.3f",
        metrics["accuracy"],
        metrics["macro_f1"],
    )

    # Save model and evaluation artifacts if requested
    if save:
        out = Path(output_dir)
        out.mkdir(exist_ok=True, parents=True)
        model_path = out / f"final_pipeline_model_{algo}.pkl"
        joblib.dump(model, model_path)
        logger.info("Model saved to: %s", model_path.name)

        output_dir = f"output/{algo}_results"
        out = Path(output_dir)
        (out / f"feature_schema_{algo}.json").write_text(
            json.dumps(FEATURE_COLUMNS, indent=2)
        )
        save_confusion_matrix_plot(y_te, y_pred, out / f"confusion_matrix_{algo}.png")
        save_classification_report_txt(y_te, y_pred, out / f"report_{algo}.txt")
        save_classification_report_json(y_te, y_pred, out / f"metrics_{algo}.json")
        logger.info("Evaluation artifacts saved to %s", out.resolve())


def get_evaluation_split(input_path: str) -> tuple:
    """Return evaluation split from the last fold of TimeSeriesSplit."""
    df = extract_features(load_data(input_path)).dropna(subset=FEATURE_COLUMNS)
    X, y = df[FEATURE_COLUMNS], df["result"]

    cv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(cv.split(X))[-1]
    X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
    return X_te, y_te


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["rf", "lgb", "cat", "stack"], default="lgb")
    p.add_argument("--input", default="data/praocessed/model_input.parquet")
    p.add_argument("--output", default="models")
    p.add_argument(
        "--save", action="store_true", help="Save model and evaluation artifacts"
    )
    args = p.parse_args()

    run_pipeline(
        algo=args.algo,
        input_path=args.input,
        output_dir=args.output,
        save=args.save,
    )
