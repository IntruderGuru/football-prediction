# ⚽ Football Match Outcome Prediction

This project predicts the outcome of football matches (Home Win / Draw / Away Win) using historical statistics, xG metrics, rolling averages, Elo ratings, and bookmaker odds.

---


## 🔧 Environment Setup

> 🐍 Python 3.11 is required. Using `conda` is recommended.

```bash
# Create conda environment
conda create -n footpred python=3.11 -y
conda activate footpred

# Install dependencies
pip install -r requirements.txt

# Optional: lock versions
# pip freeze > requirements.lock.txt

# Formatting and linting setup
pip install pre-commit ruff black
pre-commit install
```

## PYTHONPATH environment variable
PowerShell (Windows):

```bash
$env:PYTHONPATH="."
```

Linux/macOS:

```bash
export PYTHONPATH=.
```


## 📥 Data fetching and preprocessing

```bash
python scripts/fetch_football_data_uk.py
python scripts/fetch_understat.py
python scripts/merge_sources.py
```

## 📈 Model training and evaluation

```bash
python cli.py train --algo lgb --save

# Available algorithms:

#    rf — Random Forest

#    lgb — LightGBM

#    cat — CatBoost

#    stack — Stacked Ensemble (LGB + CAT + RF)

#Add --save to persist the trained model and evaluation artifacts.

```

## 📊 Evaluate a trained model

```bash
python cli.py evaluate --model-path models/final_pipeline_model_lgb.pkl
```
## 🔮 Predict single match outcome

```bash
python cli.py predict --home "Chelsea" --away "Arsenal" --date "2020-01-21" --model-path models/final_pipeline_model_lgb.pkl

```  