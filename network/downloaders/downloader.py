import pandas as pd
from abc import ABC, abstractmethod
from database.entities.leagues.league import League


class FootballDataDownloader(ABC):
    def download(self, league: League, year_start: int) -> pd.DataFrame or None:
        matches_df = self._download_csv_data(league=league, year_start=year_start)

        if matches_df is not None:
            matches_df = self._preprocess_odds(df=matches_df)
            matches_df = self._preprocess_csv_data(matches_df=matches_df, league=league)

            matches_df = matches_df.dropna().drop_duplicates()
            matches_df['Date'] = matches_df['Date'].interpolate(method='nearest')
            matches_df = matches_df.iloc[::-1].reset_index(drop=True)

        return matches_df

    def _preprocess_odds(self, df: pd.DataFrame) -> pd.DataFrame:
        home_nan_ids = df['AvgH'].isna()
        draw_nan_ids = df['AvgD'].isna()
        away_nan_ids = df['AvgA'].isna()

        if home_nan_ids.any():
            df.loc[home_nan_ids, 'AvgH'] = df.loc[home_nan_ids, 'B365H']
        if draw_nan_ids.any():
            df.loc[draw_nan_ids, 'AvgD'] = df.loc[draw_nan_ids, 'B365D']
        if away_nan_ids.any():
            df.loc[away_nan_ids, 'AvgA'] = df.loc[away_nan_ids, 'B365A']

        return df

    @abstractmethod
    def _download_csv_data(self, league: League, year_start: int) -> pd.DataFrame:
        pass

    @abstractmethod
    def _preprocess_csv_data(self, matches_df: pd.DataFrame, league: League) -> pd.DataFrame:
        pass
