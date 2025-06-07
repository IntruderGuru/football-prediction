from src.data_loader import load_data
from src.features import extract_features
from src.model import train_model, evaluate_model
from src.metrics import (
    save_classification_report_txt,
    save_classification_report_json,
    save_confusion_matrix_plot,
)
from sklearn.model_selection import train_test_split


def run_pipeline():
    # 1) load the model_input.parquet (already filtered for xG + odds)
    df = load_data("data/processed/model_input.parquet")

    # 2) compute all features (incl. rolling, temporal, form…)
    df = extract_features(df)

    # 3) drop any rows still with NA (e.g. first few matches per team)
    df = df.dropna(
        subset=[
            "xG_home",
            "xG_away",
            "bookie_prob_home",
            "bookie_prob_draw",
            "bookie_prob_away",
            # new rolled ones:
            "home_roll_xg_5",
            "away_roll_xg_5",
            "home_roll_gd_5",
            "away_roll_gd_5",
            "home_roll_form_5",
            "away_roll_form_5",
            "home_days_since",
            "away_days_since",
        ]
    )

    # 4) select your model inputs & target
    X = df[
        [
            # original
            "xG_home",
            "xG_away",
            "bookie_prob_home",
            "bookie_prob_draw",
            "bookie_prob_away",
            # new rolling
            "home_roll_xg_5",
            "away_roll_xg_5",
            "home_roll_gd_5",
            "away_roll_gd_5",
            "home_roll_form_5",
            "away_roll_form_5",
            # temporal
            "dow",
            "month",
            "home_days_since",
            "away_days_since",
        ]
    ]
    y = df["result"]

    # 5) stratified train‐test split (keep H/D/A proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # 6) train & predict
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    # 7) evaluate & save outputs
    evaluate_model(y_test, y_pred)
    save_classification_report_txt(y_test, y_pred)
    save_classification_report_json(y_test, y_pred)
    save_confusion_matrix_plot(y_test, y_pred)

    print("🚀 Pipeline finished successfully.")


if __name__ == "__main__":
    run_pipeline()
