from src.data_loader import load_data
from src.model import train_model, evaluate_model
from sklearn.model_selection import train_test_split
from src.features import extract_features
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.metrics import (
    save_classification_report_txt,
    save_classification_report_json,
    save_confusion_matrix_plot,
)


def run_pipeline():
    df = load_data()

    df = df.dropna(subset=["bookie_prob_home", "bookie_prob_draw", "bookie_prob_away"])
    df = extract_features(df)

    X = df[
        [
            "xG_home",
            "xG_away",
            "bookie_prob_home",
            "bookie_prob_draw",
            "bookie_prob_away",
        ]
    ]
    y = df["result"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    evaluate_model(y_test, y_pred)

    save_classification_report_txt(y_test, y_pred)
    save_classification_report_json(y_test, y_pred)
    save_confusion_matrix_plot(y_test, y_pred)

    print(" Pipeline finished successfully.")
