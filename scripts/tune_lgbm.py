import json
from pathlib import Path

from src.data_loader import load_data
from src.features import extract_features
from src.model import lgb_grid_search


if __name__ == "__main__":

    NUMERIC_FEATURES = [
        "xG_home",
        "xG_away",
        "bookie_prob_home",
        "bookie_prob_draw",
        "bookie_prob_away",
        "home_roll_xg_5",
        "away_roll_xg_5",
        "home_roll_gd_5",
        "away_roll_gd_5",
        "home_roll_form_5",
        "away_roll_form_5",
        "dow",
        "month",
        "home_days_since",
        "away_days_since",
    ]

    df = extract_features(load_data()).dropna()
    X = df[NUMERIC_FEATURES]
    y = df["result"]

    best_params, best_score = lgb_grid_search(X, y)
    print("Best CV score:", best_score)
    Path("output").mkdir(exist_ok=True)
    json.dump(best_params, open("output/lgb_best.json", "w"), indent=4)
    print("Saved → output/lgb_best.json")
