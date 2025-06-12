import json, joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from src.models.model import get_model

CONFIG_DIR = Path("configs")


def _read_params(path: Path):
    if path.suffix == ".json":
        return json.load(open(path))
    return joblib.load(path)


def load_params(tag: str | None):
    if tag is None:
        return _read_params(CONFIG_DIR / "lgb_default.json")
    for ext in (".json", ".pkl"):
        p = CONFIG_DIR / f"{tag}{ext}"
        if p.exists():
            return _read_params(p)
    raise FileNotFoundError(f"brak pliku configs/{tag}.json|pkl")


def build_pipeline(param_tag: str | None = None):
    params = load_params(param_tag)
    model = get_model("lgb", params=params)
    return Pipeline([("clf", model)])
