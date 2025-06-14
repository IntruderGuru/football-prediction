# Football Match Outcome Prediction

This project predicts the outcome of football matches (home win, draw, away win) based on statistical data, bookmaker odds, and historical analyses.


## 🔧 Environment setup

Python 3.11 is required. Using `conda` is recommended.

```bash
# Create environment
conda create -n footpred python=3.11 -y
conda activate footpred

# Install dependencies
pip install -r requirements.txt

# (Optional) lock versions
# pip freeze > requirements.lock.txt

# Formatting and linting
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
python -m src.pipeline --algo rf
```
