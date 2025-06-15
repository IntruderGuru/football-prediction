import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

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
    algo: str = "lgb",  # choose model: "rf", "cat", "lgb", or "stack"
    input_path: str = "data/processed/model_input.parquet",
    output_dir: str = "models",
    save: bool = False,
) -> None:
    logger.info("Loading data from %s", input_path)
    df = extract_features(load_data(input_path))
    X, y = df[FEATURE_COLUMNS], df["result"]

    print("Class counts:", y.value_counts().to_dict())

    # split data: last fold as holdout
    cv = TimeSeriesSplit(n_splits=5, gap=0)
    train_idx, test_idx = list(cv.split(X))[-1]
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    # impute missing values in training and test sets
    imputer = SimpleImputer(strategy="median").fit(X_tr)
    X_tr = pd.DataFrame(imputer.transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    X_te = pd.DataFrame(imputer.transform(X_te), columns=X_te.columns, index=X_te.index)

    logger.info(
        "Training period: %s to %s",
        df.iloc[train_idx]["date"].min(),
        df.iloc[train_idx]["date"].max(),
    )
    logger.info(
        "Test period: %s to %s",
        df.iloc[test_idx]["date"].min(),
        df.iloc[test_idx]["date"].max(),
    )

    # oversample minority classes
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_tr, y_tr)
    print("Post-SMOTE counts:", pd.Series(y_res).value_counts().to_dict())

    # train selected model
    if algo == "stack":
        base_models = [
            ("lgb", get_model(params={"class_weight": None})),
            ("cat", get_model("cat")),
            ("rf", get_model(params={"class_weight": None})),
        ]
        model = StackingClassifier(
            estimators=base_models,
            final_estimator=get_model("lgb", params={"class_weight": None}),
            passthrough=True,
            n_jobs=-1,
        ).fit(X_res, y_res)
    else:
        params = {}
        if algo == "lgb":
            tuning_file = Path("output/lgb_best.json")
            if tuning_file.exists():
                logger.info("Loading tuned LGBM parameters")
                raw = json.loads(tuning_file.read_text())

            else:
                logger.warning("No tuning file found, using defaults")
        model = train_model(X_res, y_res, algo=algo, params=params)

    # evaluate model
    y_pred = model.predict(X_te)
    metrics = evaluate_model(y_te, y_pred, X_eval=X_te, model=model, verbose=False)
    logger.info(
        "Results - Accuracy: %.3f | Macro-F1: %.3f | LogLoss: %.4f",
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["log_loss"],
    )

    # save artifacts if requested
    if save:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_file = out_dir / f"final_pipeline_model_{algo}.pkl"
        joblib.dump(model, model_file)
        logger.info("Model saved to %s", model_file)

        res_dir = Path(f"output/{algo}_results")
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "feature_schema.json").write_text(
            json.dumps(FEATURE_COLUMNS, indent=2)
        )
        save_confusion_matrix_plot(
            y_te, y_pred, res_dir / "confusion_matrix_{algo}.png"
        )
        save_classification_report_txt(y_te, y_pred, res_dir / "report_{algo}.txt")
        save_classification_report_json(y_te, y_pred, res_dir / "metrics_{algo}.json")
        logger.info("Artifacts saved to %s", res_dir.resolve())


def get_evaluation_split(input_path: str) -> tuple:
    """Return X and y for the last TimeSeriesSplit fold"""
    df = extract_features(load_data(input_path)).dropna(subset=FEATURE_COLUMNS)
    X, y = df[FEATURE_COLUMNS], df["result"]
    cv = TimeSeriesSplit(n_splits=5)
    _, test_idx = list(cv.split(X))[-1]
    return X.iloc[test_idx], y.iloc[test_idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["rf", "lgb", "cat", "stack"], default="lgb")
    parser.add_argument("--input", default="data/processed/model_input.parquet")
    parser.add_argument("--output", default="models")
    parser.add_argument("--save", action="store_true", help="Save model and metrics")
    args = parser.parse_args()

    run_pipeline(
        algo=args.algo,
        input_path=args.input,
        output_dir=args.output,
        save=args.save,
    )
