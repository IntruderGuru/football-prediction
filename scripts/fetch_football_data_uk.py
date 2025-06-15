import requests
import pandas as pd
import re
import logging
import itertools
from pathlib import Path
from typing import List, Optional
from src.constants import SEASONS, DIV_MAP, LEAGUES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
RAW_DIR = Path("data/raw/football_data_uk")
MERGED_OUT = Path("data/raw/football_data_merged.csv")


def fetch_file(season: str, code: str) -> Optional[Path]:
    url = BASE_URL.format(season=season, code=code)
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RAW_DIR / f"fd_{code}_{season}.csv"
        out_path.write_bytes(response.content)
        logger.info(f"Saved {out_path.name}")
        return out_path
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def fetch_all() -> List[Path]:
    results = []
    for season, code in itertools.product(SEASONS, LEAGUES):
        path = fetch_file(season, code)
        if path:
            results.append(path)
    return results


def parse_filename(fp: Path) -> Optional[str]:
    m = re.match(r"fd_(?P<div>[A-Z0-9]+)_(?P<season>\d{4}).csv", fp.name)
    if not m:
        logger.warning(f"Regex failed on filename: {fp.name}")
        return None
    div = m["div"].upper()
    if div not in DIV_MAP:
        logger.warning(f"Unknown league code in DIV_MAP: {div} from {fp.name}")
        return None
    return DIV_MAP[div]


def merge_fetched(files: List[Path]) -> pd.DataFrame:
    rows = []
    for fp in files:
        league = parse_filename(fp)
        if league is None:
            continue
        try:
            df = pd.read_csv(
                fp,
                usecols=[
                    "Date",
                    "HomeTeam",
                    "AwayTeam",
                    "FTHG",
                    "FTAG",
                    "B365H",
                    "B365D",
                    "B365A",
                ],
            )
            df = df.rename(
                columns={
                    "Date": "date",
                    "HomeTeam": "home_team",
                    "AwayTeam": "away_team",
                    "FTHG": "home_goals",
                    "FTAG": "away_goals",
                    "B365H": "bookie_home",
                    "B365D": "bookie_draw",
                    "B365A": "bookie_away",
                }
            )
            df["league"] = league
            df["date"] = pd.to_datetime(df["date"], dayfirst=True).dt.date
            rows.append(df)
            logger.info(f"Parsed: {fp.name}")
        except Exception as e:
            logger.warning(f"Error reading {fp.name}: {e}")
    if not rows:
        raise ValueError("No valid CSVs could be parsed.")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    fetched_files = fetch_all()
    merged_df = merge_fetched(fetched_files)
    MERGED_OUT.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(MERGED_OUT, index=False)
    logger.info(f"Saved merged CSV to {MERGED_OUT} ({merged_df.shape[0]} rows)")


if __name__ == "__main__":
    main()
