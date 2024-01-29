import pandas as pd
from abc import ABC, abstractmethod
from database.entities.leagues.league import League
from database.network.netutils import check_internet_connection


class FootballDataDownloader(ABC):
    def download(self, league: League, year_start: int) -> pd.DataFrame or None:
        matches_df = self._download_csv_data(league=league, year_start=year_start)

        if matches_df is not None:
            matches_df = self._preprocess_csv_data(matches_df=matches_df, league=league)

            matches_df.drop_duplicates(inplace=True)
            matches_df['Date'] = matches_df['Date'].interpolate(method='nearest')
            matches_df.fillna(value=-1, inplace=True)
            matches_df = matches_df.iloc[::-1].reset_index(drop=True)

        return matches_df


    @abstractmethod
    def _download_csv_data(self, league: League, year_start: int) -> pd.DataFrame:
        pass

    @abstractmethod
    def _preprocess_csv_data(self, matches_df: pd.DataFrame, league: League) -> pd.DataFrame:
        pass
