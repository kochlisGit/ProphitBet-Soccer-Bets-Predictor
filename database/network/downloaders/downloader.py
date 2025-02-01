import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from database.entities.leagues.league import League


class FootballDataDownloader(ABC):
    def download(self, league: League, year_start: int) -> pd.DataFrame or None:
        download_matches_df = self._download_csv_data(league=league, year_start=year_start)
        download_matches_df.dropna(how='all', inplace=True)

        if download_matches_df is None or download_matches_df.shape[0] == 0:
            return None

        matches_df = self._preprocess_csv_data(matches_df=download_matches_df, league=league)
        odds_df = self._extract_odds(matches_df=download_matches_df)

        matches_df.insert(loc=4, column='1', value=odds_df['1'])
        matches_df.insert(loc=5, column='X', value=odds_df['X'])
        matches_df.insert(loc=6, column='2', value=odds_df['2'])
        matches_df.drop_duplicates(inplace=True)
        matches_df['Date'] = matches_df['Date'].interpolate(method='nearest')
        matches_df.dropna(subset=['Date', 'Result'], inplace=True)
        matches_df.fillna(value=-1, inplace=True)
        matches_df = matches_df.iloc[::-1].reset_index(drop=True)
        return matches_df

    @staticmethod
    def _extract_odds(matches_df: pd.DataFrame) -> pd.DataFrame:
        def extract_odd(odd: str) -> pd.Series:
            priority_columns = [col for col in [f'AvgC{odd}', f'Avg{odd}', f'B365{odd}'] if col in matches_df.columns]
            return matches_df[priority_columns].apply(
                lambda row: row.dropna().iloc[0] if not row.dropna().empty else np.nan, axis=1
            )

        odds_df = pd.DataFrame()
        odds_df['1'] = extract_odd(odd='H')
        odds_df['X'] = extract_odd(odd='D')
        odds_df['2'] = extract_odd(odd='A')
        return odds_df

    @abstractmethod
    def _download_csv_data(self, league: League, year_start: int) -> pd.DataFrame:
        pass

    @abstractmethod
    def _preprocess_csv_data(self, matches_df: pd.DataFrame, league: League) -> pd.DataFrame:
        pass
