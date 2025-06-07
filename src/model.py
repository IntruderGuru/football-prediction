# src/model.py
from typing import Literal, Tuple, Dict, Any

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV


Algo = Literal["rf", "lgb"]


def _build_rf(params: Dict[str, Any] | None = None) -> RandomForestClassifier:
    default = dict(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    if params:
        default.update(params)
    return RandomForestClassifier(**default)


def _build_lgb(params: Dict[str, Any] | None = None) -> lgb.LGBMClassifier:
    default = dict(
        objective="multiclass",
        num_class=3,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    if params:
        default.update(params)
    return lgb.LGBMClassifier(**default)


def get_model(algo: Algo = "rf", params: Dict[str, Any] | None = None):
    if algo == "rf":
        return _build_rf(params)
    if algo == "lgb":
        return _build_lgb(params)
    raise ValueError(f"Unknown algo: {algo}")


def train_model(X, y, algo: Algo = "rf", params=None):
    model = get_model(algo, params)
    model.fit(X, y)
    return model


def evaluate_model(y_true, y_pred, verbose: bool = True) -> Dict[str, float]:
    if verbose:
        print(classification_report(y_true, y_pred))
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": (y_true == y_pred).mean(),
    }


lgb_param_grid = {
    "num_leaves": [31, 63],
    "max_depth": [-1, 8],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [300, 600],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


def lgb_grid_search(X, y, cv=3, scoring="f1_macro") -> Tuple[Dict[str, Any], float]:
    clf = _build_lgb()
    gs = GridSearchCV(
        clf,
        lgb_param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X, y)
    return gs.best_params_, gs.best_score_
