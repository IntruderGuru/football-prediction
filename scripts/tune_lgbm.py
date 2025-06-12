import json, optuna
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
import lightgbm as lgb
from src.features.build import extract_features
from src.data.loader import FootDataLoader
from src.constants import FEATURE_COLUMNS

CONFIG_DIR = Path("configs")
CONFIG_DIR.mkdir(exist_ok=True, parents=True)


def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        "max_depth": trial.suggest_int("max_depth", 1, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 400, 1200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "class_weight": {"H": 1, "A": 1, "D": 2.7},
        "objective": "multiclass",
        "num_class": 3,
        "random_state": 42,
        "n_jobs": -1,
    }

    df = extract_features(FootDataLoader().get_training_data())

    X, y = df[FEATURE_COLUMNS], df["result"]

    cv = TimeSeriesSplit(n_splits=3)
    scores = []
    for tr, te in cv.split(X):
        model = lgb.LGBMClassifier(**params, verbose=-1)
        model.fit(X.iloc[tr], y.iloc[tr])
        scores.append(f1_score(y.iloc[te], model.predict(X.iloc[te]), average="macro"))
    return sum(scores) / len(scores)


def main():
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    best_params = study.best_trial.params
    best_params.update(
        {
            "objective": "multiclass",
            "num_class": 3,
            "class_weight": {"H": 1, "A": 1, "D": 2.7},
        }
    )
    json.dump(best_params, open(CONFIG_DIR / "lgb_best.json", "w"), indent=2)
    print("💾 configs/lgb_best.json zapisany.")


if __name__ == "__main__":
    main()
