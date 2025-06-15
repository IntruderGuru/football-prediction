from typing import Literal, Any, Dict, Tuple

from src.constants import WEIGHTS_LGB, WEIGHTS_RF
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
import numpy as np

Algo = Literal["rf", "lgb", "cat"]


def _build_rf(params: Dict[str, Any] | None = None) -> RandomForestClassifier:
    """Build a Random Forest classifier with default or custom parameters."""
    cfg = dict(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight=WEIGHTS_RF,
        random_state=42,
        n_jobs=-1,
    )
    if params:
        cfg.update(params)
    return RandomForestClassifier(**cfg)


def _build_lgb(params: Dict[str, Any] | None = None) -> lgb.LGBMClassifier:
    """Build a LightGBM classifier with default or custom parameters."""
    cfg = dict(
        objective="multiclass",
        num_class=3,
        n_estimators=600,
        learning_rate=0.03,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight=WEIGHTS_LGB,
        random_state=42,
        n_jobs=-1,
    )
    if params:
        cfg.update(params)
    return lgb.LGBMClassifier(**cfg)


def _build_cat(params: Dict[str, Any] | None = None) -> CatBoostClassifier:
    """Build a CatBoost classifier with default or custom parameters."""
    cfg = dict(
        loss_function="MultiClass",
        iterations=800,
        learning_rate=0.03,
        depth=8,
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False,
    )
    if params:
        cfg.update(params)
    return CatBoostClassifier(**cfg)


def get_model(algo: Algo = "rf", params: Dict[str, Any] | None = None):
    """Return a classifier instance based on the selected algorithm."""
    if algo == "rf":
        return _build_rf(params)
    if algo == "lgb":
        return _build_lgb(params)
    if algo == "cat":
        return _build_cat(params)
    raise ValueError(f"Unknown algorithm: {algo}")


def train_model(X, y, *, algo: Algo = "rf", params=None):
    """Train the specified model on training data."""
    model = get_model(algo, params)
    model.fit(X, y)
    return model


def evaluate_model(y_true, y_pred, verbose: bool = True) -> Dict[str, float]:
    """
    Evaluate model predictions using macro F1 and accuracy.

    Handles shape mismatch between y_pred and y_true, especially when
    some models return (n, 1) arrays instead of (n,).
    """
    y_pred = np.ravel(y_pred)
    if verbose:
        print(classification_report(y_true, y_pred))
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": (y_true == y_pred).mean(),
    }


# Grid of hyperparameters for LightGBM
lgb_param_grid = {
    "num_leaves": [31, 63],
    "max_depth": [-1, 8],
    "learning_rate": [0.03, 0.05],
    "n_estimators": [500, 800],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


def lgb_grid_search(X, y, cv=3, scoring="f1_macro") -> Tuple[Dict[str, Any], float]:
    """Run GridSearchCV to tune LightGBM hyperparameters."""
    gs = GridSearchCV(
        _build_lgb(), lgb_param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )
    gs.fit(X, y)
    return gs.best_params_, gs.best_score_
