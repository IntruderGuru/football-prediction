import asyncio
import aiohttp
import pandas as pd
from understat import Understat
from pathlib import Path
from typing import List, Dict, Any
from src.constants import SEASONS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

UNDERSTAT_LEAGUE_MAP = {
    "EPL": "EPL",
    "LALIGA": "La_liga",
    "BUNDESLIGA": "Bundesliga",
    "SERIEA": "Serie_A",
    "LIGUE1": "Ligue_1",
}

LEAGUES = list(UNDERSTAT_LEAGUE_MAP.keys())


async def get_understat_data(league: str, season: int) -> List[Dict[str, Any]]:
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        data = await understat.get_league_results(
            league_name=league, season=str(season)
        )
        return data


def fetch_all_understat(leagues: List[str], seasons: List[str]) -> pd.DataFrame:
    all_dfs = []
    for league in leagues:
        understat_name = UNDERSTAT_LEAGUE_MAP[league]
        for season in seasons:
            year = int("20" + season[:2])
            logger.info(f"Fetching {league} ({understat_name}) season {year}")
            try:
                data = asyncio.run(get_understat_data(understat_name, year))
                df = pd.DataFrame(data)
                df["season"] = season
                df["league"] = league
                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to fetch {league} {season}: {e}")
    if not all_dfs:
        raise ValueError("No data fetched.")
    return pd.concat(all_dfs, ignore_index=True)


def main() -> None:
    df = fetch_all_understat(LEAGUES, SEASONS)
    df.to_csv(RAW_DIR / "understat_all.csv", index=False)
    logger.info(f"Saved Understat data to understat_all.csv ({df.shape[0]} rows)")


if __name__ == "__main__":
    main()
