import json
from pathlib import Path

import lightgbm as lgb
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

from src.constants import FEATURE_COLUMNS, WEIGHTS_LGB
from src.data_loader import load_data
from src.features import extract_features

# Configuration
N_SPLITS = 5
EARLY_STOP = 50
N_ESTIM = 2000
N_TRIALS = 150
OUT_JSON = Path("output/lgb_best.json")
OBJ = "multiclass"
USE_WEIGHTS = True


def objective(trial: optuna.Trial) -> float:
    # Sample hyperparameters
    params = dict(
        objective=OBJ,
        num_class=3,
        n_estimators=N_ESTIM,
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
        num_leaves=trial.suggest_int("num_leaves", 63, 255, log=True),
        max_depth=trial.suggest_int("max_depth", 6, 16),
        subsample=trial.suggest_float("subsample", 0.7, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 50),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 0.4),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 0.4),
        random_state=42,
        n_jobs=-1,
    )
    if USE_WEIGHTS:
        params["class_weight"] = WEIGHTS_LGB
    else:
        params.update(dict(alpha=0.75, gamma=1.5))

    f1_scores = []
    for tr_idx, val_idx in cv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        sample_w = y_tr.map(WEIGHTS_LGB).values if USE_WEIGHTS else None

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            sample_weight=sample_w,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        )
        f1_scores.append(f1_score(y_val, model.predict(X_val), average="macro"))

    return sum(f1_scores) / len(f1_scores)


if __name__ == "__main__":
    # Prepare data once in memory
    df = extract_features(load_data()).dropna(subset=FEATURE_COLUMNS)
    X = df[FEATURE_COLUMNS]
    y = df["result"]
    cv = TimeSeriesSplit(n_splits=N_SPLITS, gap=7)

    # Run Optuna with TPE
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="lgbm_macro_f1_timeseries",
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\nBest macro-F1:", round(study.best_value, 4))
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("\nSaved best parameters to", OUT_JSON.resolve())
