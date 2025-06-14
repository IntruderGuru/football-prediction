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
    algo: str = "lgb",  # "rf", "cat" albo specjalne "stack"
    input_path: str = "data/processed/model_input.parquet",
    output_dir: str = "output",
) -> None:
    logger.info("Ładowanie danych → %s", input_path)
    df = extract_features(load_data(input_path)).dropna(subset=FEATURE_COLUMNS)
    X, y = df[FEATURE_COLUMNS], df["result"]

    print("Different y classes counts:")
    print(y.value_counts())

    # TimeSeriesSplit — ostatni fold traktujemy jako test
    cv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(cv.split(X))[-1]
    X_tr, X_te, y_tr, y_te = (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    )

    print("TUU:")
    print(len(X_tr.columns.tolist()))

    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_tr, y_tr)

    # wybór modelu
    if algo == "stack":
        base = [
            ("lgb", get_model("lgb")),
            ("cat", get_model("cat")),
            ("rf", get_model("rf")),
        ]
        model = StackingClassifier(
            estimators=base,
            final_estimator=get_model("lgb"),
            passthrough=True,
            n_jobs=-1,
        ).fit(X_resampled, y_resampled)
    else:
        model = train_model(X_resampled, y_resampled, algo=algo)

    # ewaluacja
    y_pred = model.predict(X_te)
    metrics = evaluate_model(y_te, y_pred, verbose=False)
    logger.info("Acc %.3f | macro‑F1 %.3f", metrics["accuracy"], metrics["macro_f1"])

    # zapisz artefakty
    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, out / f"final_model_{algo}.pkl")
    (out / "feature_schema.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    save_confusion_matrix_plot(y_te, y_pred, out / "confusion_matrix.png")
    save_classification_report_txt(y_te, y_pred, out / "report.txt")
    save_classification_report_json(y_te, y_pred, out / "metrics.json")
    logger.info("Artefakty zapisane w %s", out.resolve())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["rf", "lgb", "cat", "stack"], default="stack")
    p.add_argument("--input", default="data/processed/model_input.parquet")
    p.add_argument("--output", default="output")
    args = p.parse_args()
    run_pipeline(algo=args.algo, input_path=args.input, output_dir=args.output)
