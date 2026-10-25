import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import List, Optional
from urllib.error import HTTPError, URLError
from tqdm import tqdm
from src.core.network.downloaders.downloader import LeagueDownloader
from src.core.leagues.league import LeagueConfig


class MainLeagueDownloader(LeagueDownloader):
    """ Downloader class for ootball-data.co.uk webpage. Utilizes leagues that are classified as "main". """

    def __init__(self):
        super().__init__()

        self._encodings = ['utf-8', 'latin1']
        self._column_mapper = {
            'HomeTeam': 'Home',
            'AwayTeam': 'Away',
            'AvgH': '1',
            'AvgD': 'X',
            'AvgA': '2',
            'FTHG': 'HG',
            'FTAG': 'AG',
            'FTR': 'Result'
        }

    def _download_dataframe(self, league_config: LeagueConfig) -> pd.DataFrame:
        def download_fn(year: int) -> Optional[pd.DataFrame]:
            """ Helper function that downloads and returns main league data in a dataframe.

                Args:
                    year (int): current year of league.
            """

            # Generating URL for the specific season.
            season_code = f'{str(year)[-2:]}{str(year + 1)[-2:]}'
            url = league_config.league.url.format(season_code)

            # Downloading data using an appropriate encoding strategy.
            for encoding in self._encodings:
                try:
                    season_df = pd.read_csv(url, encoding=encoding, on_bad_lines='skip')
                except UnicodeDecodeError:
                    continue
                except (URLError, HTTPError) as _:
                    return None

                season_df = pd.concat([season_df, pd.Series(data=year, index=season_df.index, name='Season')], axis=1)
                return season_df
            return None

        def fill_odds(data_df: pd.DataFrame, target_odds: List[str], fallback_odds: List[str]) -> pd.DataFrame:
            """ Helper function that replaces target odds with fallback odds if missing values are present.

                Steps:
                    1. Replaces missing BW with B365 for 1,X,2 and B365 with BFE for U,O.
                    2. Replaces Missing Avg with BW for 1,X,2 and Avg with B365 for U,O.
                Args
                ----
                :param data_df: (pd.DataFrame) dataframe containing league data.
                :param target_odds: (List[str]) list of target odds to be replaced.
                :param fallback_odds: (List[str]) list of fallback odds to replace with.
            """

            missing_ids = data_df[target_odds].isna().any(axis=1)
            if missing_ids.any():
                data_df.loc[missing_ids, target_odds] = data_df.loc[missing_ids, fallback_odds].values
            return data_df

        # Computing max concurrent workers.
        years = range(league_config.start_year, date.today().year + 1)
        max_workers = min(len(years), max(1, os.cpu_count() - 1))

        # Iterating seasons concurrently and downloading a dataframe per season.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            dfs_list = list(
                tqdm(
                    executor.map(download_fn, years),
                    total=len(years),
                    desc='Downloading Year'
                )
            )
        dfs_list = [df for df in dfs_list if df is not None]

        if not dfs_list:
            raise AssertionError(
                f'\nFailed to download league: {league_config.league} from year {league_config.start_year}.'
            )

        # Concatenating all dataframes.
        df = pd.concat(dfs_list, axis=0, ignore_index=True)

        # Filling missing odds.
        df = fill_odds(data_df=df, target_odds=['BWH', 'BWD', 'BWA'], fallback_odds=['B365H', 'B365D', 'B365A'])
        df = fill_odds(data_df=df, target_odds=['AvgH', 'AvgD', 'AvgA'], fallback_odds=['BWH', 'BWD', 'BWA'])

        # Renaming columns.
        df = df.rename(columns=self._column_mapper)
        return df
