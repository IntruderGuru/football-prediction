import json
from pathlib import Path
import joblib

from src.data_loader import load_data
from src.features import extract_features
from src.model import lgb_grid_search, train_model
from src.constants import FEATURE_COLUMNS


if __name__ == "__main__":

    df = extract_features(load_data()).dropna(subset=FEATURE_COLUMNS + ["result"])
    X = df[FEATURE_COLUMNS]
    y = df["result"]

    print("🔍 Running grid search for LGBM...")
    best_params, best_score = lgb_grid_search(X, y, cv=5)
    print(f"Best CV macro-F1: {best_score:.3f}")
    print("Best hyperparameters:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "lgb_best.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("Saved hyperparameters → output/lgb_best.json")

    print("Training final model with best parameters...")
    model = train_model(X, y, algo="lgb", params=best_params)

    joblib.dump(model, output_dir / "final_model_lgb.pkl")
    with open(output_dir / "feature_schema.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    print("Saved trained model → output/final_model_lgb.pkl")
