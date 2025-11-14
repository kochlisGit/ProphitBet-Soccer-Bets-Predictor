import logging
import pandas as pd
from datetime import date
from typing import Optional, List
from urllib.error import HTTPError
from tqdm import tqdm
from src.network.leagues.downloaders.downloader import FootballDataDownloader
from src.network.leagues.league import League


class MainLeagueDownloader(FootballDataDownloader):
    """ Downloader class for leagues classified as "main". """

    def __init__(self):
        super().__init__()

        self._columns = [
            'Date',         # Match date.
            'Season',       # League's current season.
            'HomeTeam',     # Name of the home team.
            'AwayTeam',     # Name of the away team.
            'FTHG',         # Full-time scored goals by home team.
            'FTAG',         # Full-time scored goals by away team.
            'FTR',          # Full-time result (H/A/D).
            'AvgH',         # The average booking odd for home win.
            'AvgD',         # The average booking odd for draw.
            'AvgA'          # The average booking odd for away win.
        ] + self._get_additional_columns()

    def _get_additional_columns(self) -> List[str]:
        return [
            'HST',          # Number of shots on target by home team.
            'AST',          # Number of shots on target by away team.
            'HC',           # Number of kicked corners by home team.
            'AC'            # Number of kicked corners by away team.
        ]

    def _download_dataframe(self, league: League, start_year: int) -> Optional[pd.DataFrame]:
        """ Download csv data for each season (year, year + 1) and concatenate the data. """

        def download_fn(year: int):
            url = league.url.format(f'{str(year)[-2:]}{str(year + 1)[-2:]}')
            try:
                df = pd.read_csv(url)
            except HTTPError as http_error:
                if year < date.today().year:
                    logging.info(
                        f'{http_error}\n'
                        f'Failed to download from: {url} from league: {str(league)}, Year = {year}.'
                    )
                return None
            except UnicodeDecodeError as _:
                df = pd.read_csv(url, encoding='latin1')

            df['Season'] = year
            return df

        dfs_list = []
        for current_year in tqdm(iterable=[year for year in range(start_year, date.today().year + 1)], desc='Downloading Year'):
            df = download_fn(year=current_year)

            if df is not None:
                dfs_list.append(df)

        if not dfs_list:
            return None

        df = pd.concat(dfs_list, axis=0)
        return df

    def _preprocess_dataframe(self, df: pd.DataFrame, start_year: int) -> pd.DataFrame:
        """ Fetches the specified dataframe columns and renames them to match the base class expected columns. """

        # Fill missing average odd values with B365 odds.
        missing_ids = df['AvgH'].isna()
        if missing_ids.any():
            b365_odds = df.loc[missing_ids, ['B365H', 'B365D', 'B365A']].values
            df.loc[missing_ids, ['AvgH', 'AvgD', 'AvgA']] = b365_odds

        missing_ids = df['Avg>2.5'].isna()
        if missing_ids.any():
            b365_odds = df.loc[missing_ids, ['B365>2.5', 'B365<2.5']].values
            df.loc[missing_ids, ['Avg>2.5', 'Avg<2.5']] = b365_odds

        # Rename columns.
        df = df[self._columns]
        df = df.rename(columns={
            'HomeTeam': 'Home',
            'AwayTeam': 'Away',
            'AvgH': '1',
            'AvgD': 'X',
            'AvgA': '2',
            'FTHG': 'HG',
            'FTAG': 'AG',
            'FTR': 'Result'
        })
        return df
