import pandas as pd
from database.entities.leagues.league import League
from database.network.downloaders.downloader import FootballDataDownloader


class ExtraLeagueDownloader(FootballDataDownloader):
    def _download_csv_data(self, league: League, year_start: int) -> pd.DataFrame or None:
        try:
            matches_df = pd.read_csv(league.data_url)
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

        columns = ['Date', 'Season', 'Home', 'Away', 'AvgH', 'AvgD', 'AvgA', 'HG', 'AG', 'Res']
        columns_mapper = {
            'Home': 'Home Team',
            'Away': 'Away Team',
            'AvgH': '1',
            'AvgD': 'X',
            'AvgA': '2',
            'Res': 'Result'
        }
        matches_df['Season'] = matches_df['Season'].apply(set_season)
        return matches_df[columns].rename(columns=columns_mapper)
