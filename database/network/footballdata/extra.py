import pandas as pd
from website.models import AvailableLeague
from database.network.footballdata.api import FootballDataAPI


class ExtraLeagueAPI(FootballDataAPI):
    def _download(self, league: AvailableLeague) -> pd.DataFrame:
        return pd.read_csv(league.url)

    def _process_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        matches_df = matches_df[
            [
                "Date",
                "Season",
                "Home",
                "Away",
                "AvgH",
                "AvgD",
                "AvgA",
                "HG",
                "AG",
                "Res",
            ]
        ]
        matches_df = matches_df.rename(
            columns={
                "Home": "Home Team",
                "Away": "Away Team",
                "AvgH": "1",
                "AvgD": "X",
                "AvgA": "2",
                "Res": "Result",
            }
        )
        return matches_df
