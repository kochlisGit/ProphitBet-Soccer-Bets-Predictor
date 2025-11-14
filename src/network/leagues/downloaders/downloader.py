import pandas as pd
from abc import ABC, abstractmethod
from src.network.leagues.league import League
from typing import List, Optional


class FootballDataDownloader(ABC):
    """ Base class that downloads and processes leagues from football-data webpage. """

    def __init__(self):
        self._expected_columns = [
            'Date',
            'Season',
            'Home',
            'Away',
            'HG',
            'AG',
            'Result',
            '1',
            'X',
            '2'
        ] + self._get_additional_columns()
        self._time_format = '%Y-%m-%d'

    @property
    def expected_columns(self) -> List[str]:
        return self._expected_columns

    @abstractmethod
    def _get_additional_columns(self) -> List[str]:
        pass

    @abstractmethod
    def _download_dataframe(self, league: League, start_year: int) -> Optional[pd.DataFrame]:
        pass

    @abstractmethod
    def _preprocess_dataframe(self, df: pd.DataFrame, start_year: int) -> pd.DataFrame:
        pass

    def download(self, league: League, start_year: int) -> Optional[pd.DataFrame]:
        """ Downloads, preprocesses and validates the requested league data.
            Returns a dataframe with the collected data.
        """

        if start_year < league.start_year:
            raise ValueError(
                f'Cannot download from year {start_year}. The minimum year for {str(league)} is {league.start_year}.'
            )

        df = self._download_dataframe(league=league, start_year=start_year)

        if df.shape[0] == 0 or df is None:
            return None

        # Preprocess the downloaded dataframe.
        df = self._preprocess(df=df, start_year=start_year)

        return df

    def _preprocess(self, df: pd.DataFrame, start_year: int) -> Optional[pd.DataFrame]:
        """ Preprocesses the dataframe by:
            1) Formatting the columns (should be handled in _preprocess_dataframe function).
            2) Validates the dataframe.
            3) Drops NULL dates or matches without a registered result.
            4) Sorts matches by date in ascending order.
            5) Drops duplicated matches.
            6) Computes Result-U/O column (useful for U/O Classification tasks).
            7) Computes match Week.
         """

        # Format the downloaded dataframe columns.
        df = self._preprocess_dataframe(df=df, start_year=start_year)

        # Validate the dataframe.
        if df.columns.tolist() != self._expected_columns:
            raise ValueError(f'Expected columns: {self._expected_columns} got {df.columns.to_list()}')
        if df['Season'].min() < start_year:
            raise ValueError(f'The minimum downloaded year should be {start_year}, got {df["Date"].min()}.')

        # Drop NULL values, validate that only 3 results exist
        df = df.dropna(subset=['Date', 'Result'])

        if df['Result'].nunique(dropna=False) > 3:
            raise ValueError(f'Expected "Result" columns to be one of <H, D, A>, got {df["Result"].unique().tolist()}')

        # sort matches by date and drop the duplicates.
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, utc=True, format='mixed').dt.strftime(self._time_format)
        df = df.sort_values(by=['Date', 'Home'], ascending=True)
        df = df.drop_duplicates()

        if df.shape[0] == 0 or df is None:
            return None

        # Add Result-U/O column.
        result_uo_index = df.columns.tolist().index('Result') + 1
        result_uo = (df['HG'] + df['AG']).ge(2.5).replace({True: 'O', False: 'U'})
        df.insert(loc=result_uo_index, column='Result-U/O', value=result_uo)

        # Compute game of week.
        df['Week'] = df.groupby(by=['Season', 'Home']).cumcount() + 1
        return df
