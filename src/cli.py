import typer
from pathlib import Path

from scripts.fetch_football_data_uk import main as fetch_fd
from scripts.fetch_understat import main as fetch_us
from scripts.merge_sources import merge_sources
from scripts.model_predict import main as predict_main
from scripts.tune_lgbm import main as tune_lgbm_main
from scripts.run_pipeline import train as train_pipeline

app = typer.Typer(help="Football Match Prediction CLI")


@app.command()
def fetch():
    fetch_fd()
    fetch_us()


@app.command()
def merge():
    merge_sources()


@app.command()
def train(params: str = typer.Option("lgb_best", help="Tag z katalogu configs/")):
    train_pipeline(params=params)


@app.command()
def predict(
    model_path: Path = typer.Option(
        "output/model_lgb_lgb_best.pkl", help="Ścieżka do modelu"
    ),
    schema_path: Path = typer.Option(
        "output/feature_schema.json", help="Ścieżka do schematu"
    ),
    features: str = typer.Option(..., help="Inline JSON, np. '{\"xG_home\":1.4, ...}'"),
):
    """Predykcja pojedynczego meczu"""
    import sys

    sys.argv = [
        "model_predict.py",
        "--model",
        str(model_path),
        "--schema",
        str(schema_path),
        "--features",
        features,
    ]
    predict_main()


@app.command("predict-match")
def predict_match_cli(
    home_team: str = typer.Option(..., help="Nazwa gospodarzy"),
    away_team: str = typer.Option(..., help="Nazwa gości"),
    date: str = typer.Option(..., help="Data w formacie YYYY-MM-DD"),
):
    from scripts.infer_match import predict_match
    import json

    result = predict_match(home_team, away_team, date)
    print(json.dumps(result, indent=2))


@app.command()
def tune():
    """Optymalizacja hiperparametrów modelu LightGBM"""
    tune_lgbm_main()


if __name__ == "__main__":
    app()
