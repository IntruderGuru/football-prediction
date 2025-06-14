import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.data_loader import load_data
from src.features import extract_features
from src.constants import FEATURE_COLUMNS

df = extract_features(load_data()).dropna(subset=FEATURE_COLUMNS + ["result"])
X = df[FEATURE_COLUMNS]
y = df["result"]

tscv = TimeSeriesSplit(n_splits=5)
_, test_idx = list(tscv.split(X))[-1]
X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

model_path = Path("output/final_model_lgb.pkl")
model = joblib.load(model_path)

y_pred = model.predict(X_test)

print("Honest Evaluation on time-based holdout:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"✅ Macro F1: {f1_score(y_test, y_pred, average='macro'):.3f}")
