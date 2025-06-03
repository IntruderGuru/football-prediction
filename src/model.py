from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_test, y_pred):
    print("Model evaluation:\n")
    print(classification_report(y_test, y_pred))
