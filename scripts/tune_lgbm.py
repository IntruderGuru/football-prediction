import json
from pathlib import Path

from src.data_loader import load_data
from src.features import extract_features
from src.model import lgb_grid_search
from src.constants import FEATURE_COLUMNS


if __name__ == "__main__":

    df = extract_features(load_data()).dropna()
    X = df[FEATURE_COLUMNS]
    y = df["result"]

    best_params, best_score = lgb_grid_search(X, y)
    print("Best CV score:", best_score)
    Path("output").mkdir(exist_ok=True)
    json.dump(best_params, open("output/lgb_best.json", "w"), indent=4)
    print("Saved → output/lgb_best.json")
