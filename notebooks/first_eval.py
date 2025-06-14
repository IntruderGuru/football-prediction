import joblib
from sklearn.metrics import classification_report, f1_score, accuracy_score
from src.data_loader import load_data
from src.features import extract_features
from src.constants import FEATURE_COLUMNS
from sklearn.model_selection import TimeSeriesSplit

df = extract_features(load_data()).dropna(subset=FEATURE_COLUMNS + ["result"])
X = df[FEATURE_COLUMNS]
y = df["result"]

tscv = TimeSeriesSplit(n_splits=5)
train_idx, test_idx = list(tscv.split(X))[-1]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

model = joblib.load("output/final_model_lgb.pkl")

y_pred = model.predict(X_test)

print("🔍 Evaluation of final tuned model:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"✅ Macro F1: {f1_score(y_test, y_pred, average='macro'):.3f}")
