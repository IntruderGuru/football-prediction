# scripts/model_predict.py
import argparse
import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd


# ──────────────────── helpery ─────────────────────
def load_schema(schema_path: str) -> list[str]:
    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ Brak pliku schematu kolumn: {path}")
    return json.loads(path.read_text())


def build_feature_df(raw_json: str | Path, schema: list[str]) -> pd.DataFrame:
    """Z surowego JSON-a (inline lub plik) → DataFrame w kolejności schema."""
    # odczyt
    feats = (
        json.loads(raw_json.read_text())
        if isinstance(raw_json, Path)
        else json.loads(raw_json)
    )

    # walidacja ➜ brakujące kolumny
    missing = [c for c in schema if c not in feats]
    if missing:
        raise ValueError(f"Brakuje feature’ów: {missing}")

    return pd.DataFrame([[feats[c] for c in schema]], columns=schema)


# ────────────────────────── główna funkcja ───────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Inferencja - pojedynczy mecz")
    ap.add_argument("--model", default="output/final_model_lgb.pkl")
    ap.add_argument("--schema", default="output/feature_schema.json")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--features", help="inline JSON, np. '{\"xG_home\":1.4, …}'")

    g.add_argument("--features-file", help="plik JSON z feature’ami")

    args = ap.parse_args()

    # ① ładujemy artefakty
    schema = load_schema(args.schema)
    model = joblib.load(args.model)

    # ② przygotowujemy wektor cech
    X = (
        build_feature_df(Path(args.features_file), schema)
        if args.features_file
        else build_feature_df(args.features, schema)
    )

    # ③ predykcja
    proba = model.predict_proba(X)[0]
    pred = model.classes_[np.argmax(proba)]

    # ④ wynik
    print(f"Prediction  : {pred}")
    print(f"Probabilities: {dict(zip(model.classes_, proba.round(3)))}")


if __name__ == "__main__":  # pragma: no cover
    main()
