import pandas as pd
from datetime import date
from database.entities.leagues.league import League
from database.network.downloaders.downloader import FootballDataDownloader


class MainLeagueDownloader(FootballDataDownloader):
    def _download_csv_data(self, league: League, year_start: int) -> pd.DataFrame or None:
        def download(year: int) -> pd.DataFrame:
            url = league.data_url.format(f'{str(year)[-2:]}{str(year + 1)[-2:]}')

            try:
                matches_df = pd.read_csv(url)
                matches_df['Season'] = year
                return matches_df
            except Exception as e:
                if year < date.today().year:
                    print(
                        f'{e}\nWarning: Failed to download file: {url} '
                        f'from league: {league.country}: {league.name}, Year = {year}'
                    )

                return pd.DataFrame()

        dfs_list = list(map(download, [year for year in range(year_start, date.today().year + 1)]))
        matches_df = pd.concat(dfs_list, axis=0)
        return matches_df if matches_df.shape[0] > 0 else None

    def _preprocess_csv_data(self, matches_df: pd.DataFrame, league: League) -> pd.DataFrame:
        columns = ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'AvgH', 'AvgD', 'AvgA', 'FTHG', 'FTAG', 'FTR']
        columns_mapper = {
            'HomeTeam': 'Home Team',
            'AwayTeam': 'Away Team',
            'AvgH': '1',
            'AvgD': 'X',
            'AvgA': '2',
            'FTHG': 'HG',
            'FTAG': 'AG',
            'FTR': 'Result'
        }
        return matches_df[columns].rename(columns=columns_mapper)
