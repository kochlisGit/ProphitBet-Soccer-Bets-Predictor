import pandas as pd
from abc import ABC, abstractmethod
from datetime import date
from database.entities.league import League


class FootballDataAPI(ABC):
    def download(self, league: League) -> pd.DataFrame:
        matches_df = self._download(league=league)
        matches_df = self._process_features(matches_df=matches_df)
        matches_df = matches_df.drop_duplicates()
        matches_df = matches_df.iloc[::-1].reset_index(drop=True)
        return matches_df

    @abstractmethod
    def _download(self, league: League) -> pd.DataFrame:
        pass

    @abstractmethod
    def _process_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        pass
