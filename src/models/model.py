from typing import Literal, Any, Dict
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from src.constants import WEIGHTS_RF, WEIGHTS_LGB

Algo = Literal["rf", "lgb", "cat"]


def _build_rf(params: Dict[str, Any] | None = None):
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


def _build_lgb(params: Dict[str, Any] | None = None):
    if not params:
        params = {}
    base = dict(
        objective="multiclass",
        num_class=3,
        class_weight=WEIGHTS_LGB,
        random_state=42,
        n_jobs=-1,
    )
    base.update(params)
    return lgb.LGBMClassifier(**base, verbose=-1)


def _build_cat(params: Dict[str, Any] | None = None):
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
    return {"rf": _build_rf, "lgb": _build_lgb, "cat": _build_cat}[algo](params)
