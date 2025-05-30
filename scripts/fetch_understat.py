import asyncio
from understat import Understat
import pandas as pd
import aiohttp


async def get_understat_data(league="EPL", season=2022):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        data = await understat.get_league_results(
            league_name=league, season=str(season)
        )

        return data


def run(league="EPL", start=2019, end=2024):
    all_data = []
    for season in range(start, end + 1):
        print(f"Fetching {league} season {season}")
        data = asyncio.run(get_understat_data(league, season))
        df = pd.DataFrame(data)
        df["season"] = season
        all_data.append(df)
    full_df = pd.concat(all_data)
    return full_df


if __name__ == "__main__":
    df = run("EPL", 2019, 2024)
    df.to_csv("data/raw/understat_epl.csv", index=False)
    print("Saved to data/raw/understat_epl.csv")
