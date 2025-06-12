import json
import typer
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from joblib import dump

from src.data.loader import FootDataLoader
from src.features.build import extract_features
from src.pipeline import build_pipeline
from src.constants import FEATURE_COLUMNS

app = typer.Typer()


@app.command()
def fetch(force: bool = False):
    FootDataLoader().build(force=force)


@app.command()
def train(params: str = None, outdir: Path = Path("output")):

    df = extract_features(FootDataLoader().get_training_data())

    X, y = df[FEATURE_COLUMNS], df["result"]

    cv = TimeSeriesSplit(n_splits=5)
    tr, te = list(cv.split(X))[-1]
    pipe = build_pipeline(params).fit(X.iloc[tr], y.iloc[tr])

    y_pred = pipe.predict(X.iloc[te])
    print(classification_report(y.iloc[te], y_pred))

    outdir.mkdir(exist_ok=True, parents=True)
    dump(pipe, outdir / f"model_lgb_{params or 'default'}.pkl")
    (outdir / "feature_schema.json").write_text(json.dumps(FEATURE_COLUMNS))
    print("✅ zapisano model i schemat cech")


if __name__ == "__main__":
    app()
