import argparse
import json
import joblib
import ast
from pathlib import Path

import numpy as np
import pandas as pd


def load_schema(schema_path: str) -> list[str]:
    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ Brak pliku schematu kolumn: {path}")
    return json.loads(path.read_text())


def build_feature_df(raw_json: str | Path, schema: list[str]) -> pd.DataFrame:

    try:
        if isinstance(raw_json, Path):

            txt = raw_json.read_text(encoding="utf-8-sig")
        else:
            txt = raw_json
        txt = txt.strip()
        if (txt.startswith("'") and txt.endswith("'")) or (
            txt.startswith('"') and txt.endswith('"')
        ):
            txt = txt[1:-1]
        try:
            feats = json.loads(txt)
        except json.JSONDecodeError as e:
            feats = ast.literal_eval(txt)
    except Exception as e:
        raise ValueError(f"❌ Błąd podczas parsowania cech: {e}")

    missing = [c for c in schema if c not in feats]
    if missing:
        raise ValueError(f"Brakuje feature’ów: {missing}")

    return pd.DataFrame([[feats[c] for c in schema]], columns=schema)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inferencja - pojedynczy mecz")
    ap.add_argument("--model", default="output/final_model_lgb.pkl")
    ap.add_argument("--schema", default="output/feature_schema.json")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--features",
        help=(
            "inline JSON z cechami, np. "
            '\'{"bookie_prob_home":0.45,"away_roll_xg_5":0.1, ...}\''
        ),
    )

    g.add_argument("--features-file", help="plik JSON z feature’ami")

    args = ap.parse_args()

    schema = load_schema(args.schema)
    model = joblib.load(args.model)

    X = (
        build_feature_df(Path(args.features_file), schema)
        if args.features_file
        else build_feature_df(args.features, schema)
    )

    proba = model.predict_proba(X)[0]
    pred = model.classes_[np.argmax(proba)]

    print(f"Prediction  : {pred}")
    print(f"Probabilities: {dict(zip(model.classes_, proba.round(3)))}")


if __name__ == "__main__":
    main()
