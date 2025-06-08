from typing import Literal, Any, Dict, Tuple

from src.constants import WEIGHTS_LGB, WEIGHTS_RF
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

Algo = Literal["rf", "lgb", "cat", "xgb"]


def _build_xgb(params: Dict[str, Any] | None = None) -> XGBClassifier:
    cfg = dict(
        objective="multi:softprob",
        num_class=3,
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        use_label_encoder=False,
    )
    if params:
        cfg.update(params)
    return XGBClassifier(**cfg)


def _build_rf(params: Dict[str, Any] | None = None) -> RandomForestClassifier:
    cfg = dict(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        class_weight=WEIGHTS_RF,
        random_state=42,
        n_jobs=-1,
    )
    if params:
        cfg.update(params)
    return RandomForestClassifier(**cfg)


def _build_lgb(params: Dict[str, Any] | None = None) -> lgb.LGBMClassifier:
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
    if algo == "rf":
        return _build_rf(params)
    if algo == "lgb":
        return _build_lgb(params)
    if algo == "cat":
        return _build_cat(params)
    if algo == "xgb":
        return _build_xgb(params)
    raise ValueError(f"Unknown algo: {algo}")


def train_model(X, y, *, algo: Algo = "rf", params=None):
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
    "learning_rate": [0.03, 0.05],
    "n_estimators": [500, 800],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


def lgb_grid_search(X, y, cv=3, scoring="f1_macro") -> Tuple[Dict[str, Any], float]:
    gs = GridSearchCV(
        _build_lgb(), lgb_param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )
    gs.fit(X, y)
    return gs.best_params_, gs.best_score_
