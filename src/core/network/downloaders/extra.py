import pandas as pd
from urllib.error import HTTPError, URLError
from src.core.network.downloaders.downloader import LeagueDownloader
from src.core.leagues.league import LeagueConfig


class ExtraLeagueDownloader(LeagueDownloader):
    """ Downloader class for ootball-data.co.uk webpage. Utilizes leagues that are classified as "extra". """

    def __init__(self):
        super().__init__()

        self._column_mapper = {
            'AvgCH': '1',
            'AvgCD': 'X',
            'AvgCA': '2',
            'PSCH': 'BWH',
            'PSCD': 'BWD',
            'PSCA': 'BWA',
            'B365CH': 'B365H',
            'B365CD': 'B365D',
            'B365CA': 'B365A',
            'Res': 'Result'
        }

    def _download_dataframe(self, league_config: LeagueConfig) -> pd.DataFrame:
        def format_season(season: str) -> int:
            """ Sets a fixed season format: integer with 4 digits.

                Args
                ----
                :param season: (str | int) League's season value.
            """

            if len(season) == 4:
                return int(season)
            elif len(season) == 9:
                return int(season.split('/')[0])
            else:
                raise ValueError(
                    f'Season is expected to be a string with 4 digits "yyyy" or 9 digits "yyyy/yyyy+1", got "{season}".'
                )

        def patch_usa_league(data_df: pd.DataFrame) -> pd.DataFrame:
            """ Fixes USA's odds for home win and convert it into float.

                Args
                ----
                :param data_df: (pd.DataFram) USA's dataframe.
            """

            row_id = data_df['AvgCH'] == 'x'
            data_df.loc[row_id, 'AvgCH'] = data_df.loc[row_id, 'PSCH'].astype(str).values
            data_df['AvgCH'] = df['AvgCH'].astype(float)
            return data_df

        league = league_config.league
        try:
            df = pd.read_csv(league.url, on_bad_lines='skip')
        except (URLError, HTTPError) as e:
            raise URLError(
                f'\n{e}\nFailed to download league {league_config.league}, due to invalid url: "{league.url}"'
            )

        # Formatting season's column.
        if not pd.api.types.is_integer_dtype(df['Season']):
            df['Season'] = df['Season'].apply(format_season)

        # Patch USA's odd error.
        if league.country == 'USA':
            df = patch_usa_league(data_df=df)

        # Rename columns.
        df = df.rename(columns=self._column_mapper)
        return df
