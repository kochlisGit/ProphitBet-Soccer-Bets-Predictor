from database.entities.leagues import League
import pandas as pd


class FootballDataLeaguesAPI:
    def download_league(self, league: League, download_directory: str) -> bool:
        if league.league_type == 'main':
            season_dfs = self._download_main_league(league=league)
        elif league.league_type == 'extra':
            season_dfs = self._download_extra_league(league=league)
        else:
            season_dfs = None

        if season_dfs is not None:
            self._store_seasons(
                season_df=season_dfs,
                download_directory=download_directory,
                year_start=league.year_start
            )
            return True
        else:
            return False

    def _download_main_league(self, league: League) -> list:
        league_dfs = []

        for url in league.url_list:
            try:
                df = pd.read_csv(url)
            except:
                df = None

            if df is not None:
                league_df = self._convert_main_to_basic_league(df=df)

                if pd.DatetimeIndex(league_df.head(1)['Date'], dayfirst=True) < \
                        pd.DatetimeIndex(league_df.tail(1)['Date'], dayfirst=True):
                    league_df = league_df.iloc[::-1]

                league_dfs.append(league_df)
        return league_dfs

    def _download_extra_league(self, league: League) -> list:
        league_df = pd.read_csv(league.base_url)
        league_df = self._convert_extra_to_basic_league(df=league_df)

        if pd.DatetimeIndex(league_df.head(1)['Date'], dayfirst=True) < \
                pd.DatetimeIndex(league_df.tail(1)['Date'], dayfirst=True):
            league_df = league_df.iloc[::-1]

        return [
            league_df[league_df['Season'] == season]
            for season in range(league.year_start, league_df['Season'].unique().max() + 1)
        ]

    @staticmethod
    def _convert_extra_to_basic_league(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(
            columns={
                'Home': 'Home Team', 'Away': 'Away Team', 'Res': 'Result',
                'AvgH': '1', 'AvgA': '2', 'AvgD': 'X'
            }
        )

        df[df['HG'].isna()] = 0
        df[df['AG'].isna()] = 0
        df['HG'] = df['HG'].astype(int)
        df['AG'] = df['AG'].astype(int)
        return df

    @staticmethod
    def _convert_main_to_basic_league(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(
            columns={
                'HomeTeam': 'Home Team', 'AwayTeam': 'Away Team',
                'FTR': 'Result', 'FTHG': 'HG', 'FTAG': 'AG',
                'B365H': '1', 'B365A': '2', 'B365D': 'X'
            }
        )
        return df

    @staticmethod
    def _store_seasons(season_df: list, download_directory: str, year_start: int):
        for i, df in enumerate(season_df):
            download_filepath = f'{download_directory}/{year_start + i}.csv'
            df.to_csv(download_filepath)

    def update_league(self, league: League, download_directory: str):
        if league.league_type == 'main':
            url_list = league.url_list
            league_df = pd.read_csv(url_list[-1])
            league_df = self._convert_main_to_basic_league(df=league_df)

            if pd.DatetimeIndex(league_df.head(1)['Date'], dayfirst=True) < \
                    pd.DatetimeIndex(league_df.tail(1)['Date'], dayfirst=True):
                league_df = league_df.iloc[::-1]

            download_filepath = f'{download_directory}/{league.year_start + len(url_list) - 1}.csv'
            league_df.to_csv(download_filepath)
        else:
            self.download_league(league=league, download_directory=download_directory)
