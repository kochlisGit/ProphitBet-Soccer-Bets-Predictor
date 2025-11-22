import logging
import pandas as pd
from typing import List, Optional
from urllib.error import URLError
from src.network.leagues.downloaders.downloader import FootballDataDownloader
from src.network.leagues.league import League


class ExtraLeagueDownloader(FootballDataDownloader):
    """ Downloader class for leagues classified as "extra". """

    def __init__(self):
        super().__init__()

        self._columns = [
            'Date',         # Match date.
            'Season',       # League's current season.
            'Home',         # Name of the home team.
            'Away',         # Name of the away team.
            'HG',           # Full-time scored goals by home team.
            'AG',           # Full-time scored goals by away team.
            'Res',          # Full-time result (H/A/D).
            'AvgCH',        # The average booking odd for home win.
            'AvgCD',        # The average booking odd for draw.
            'AvgCA'         # The average booking odd for away win.
        ]

    def _get_additional_columns(self) -> List[str]:
        return []

    def _download_dataframe(self, league: League, start_year: int) -> Optional[pd.DataFrame]:
        """ Downloads and returns csv data for all seasons. """

        try:
            return pd.read_csv(league.url, on_bad_lines='skip')
        except URLError as e:
            logging.info(f'{e}\nFailed to download from: {league.url} from league: {str(league)}.')
            return None

    def _preprocess_dataframe(self, df: pd.DataFrame, start_year: int) -> pd.DataFrame:
        """ Fetches and renames the specified dataframe columns and samples, and rectifies Season format. """

        def format_season(season: str) -> int:
            if len(season) == 4:
                return int(season)
            elif len(season) == 9:
                return int(season.split('/')[0])
            else:
                raise NotImplementedError(f'Invalid season: {season}')

        # Format seasons (e.g. 2010/2012) to the specified format (e.g. 2010).
        if df['Season'].dtype != int:
            df['Season'] = df['Season'].apply(format_season).astype(int)

        # Fetch matches of season higher than start year.
        df = df[df['Season'] >= start_year]

        # Fetch and rename columns.
        df = df[self._columns]
        df = df.rename(columns={
            'AvgCH': '1',
            'AvgCD': 'X',
            'AvgCA': '2',
            'Res': 'Result'
        })
        return df
