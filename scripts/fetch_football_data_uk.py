import requests, itertools
from pathlib import Path

SEASONS = ["2324", "2223", "2122", "2021", "1920"]
LEAGUES = {
    "E0": "EPL",
    "SP1": "LALIGA",
    "D1": "BUNDESLIGA",
    "I1": "SERIEA",
    "F1": "LIGUE1",
}
BASE = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
OUT_DIR = Path("data/raw")


def fetch_file(season: str, code: str) -> None:
    url = BASE.format(season=season, code=code)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"fd_{code}_{season}.csv").write_bytes(resp.content)


def main() -> None:
    for season, code in itertools.product(SEASONS, LEAGUES):
        try:
            fetch_file(season, code)
        except Exception:
            pass


if __name__ == "__main__":
    main()
