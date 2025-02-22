import requests
import pandas as pd
from io import StringIO
from database.entities.leagues.league import League
from database.network.downloaders.downloader import FootballDataDownloader


class ExtraLeagueDownloader(FootballDataDownloader):
    def _download_csv_data(self, league: League, year_start: int) -> pd.DataFrame or None:
        try:
            response = requests.get(league.data_url, verify=False)
            response.raise_for_status()
            matches_df = pd.read_csv(StringIO(response.text))
        except Exception as e:
            print(e)

            return None

        matches_df['Season'] = matches_df['Season'].astype(str)
        return matches_df[matches_df['Season'] >= str(year_start)]

    def _preprocess_csv_data(self, matches_df: pd.DataFrame, league: League) -> pd.DataFrame:
        def set_season(season: str) -> int:
            if len(season) == 4:
                return int(season)
            elif len(season) == 9:
                return int(season.split('/')[0])
            else:
                raise NotImplementedError(f'Invalid season: {season}')

        columns = ['Date', 'Season', 'Home', 'Away', 'HG', 'AG', 'Res']
        columns_mapper = {
            'Home': 'Home Team',
            'Away': 'Away Team',
            'Res': 'Result'
        }
        matches_df['Season'] = matches_df['Season'].apply(set_season)
        return matches_df[columns].rename(columns=columns_mapper)
