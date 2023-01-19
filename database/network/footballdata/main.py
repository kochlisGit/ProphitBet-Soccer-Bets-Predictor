import pandas as pd
from datetime import date
from database.entities.league import League
from database.network.footballdata.api import FootballDataAPI


class MainLeagueAPI(FootballDataAPI):
    def _download(self, league: League) -> pd.DataFrame:
        url_list = self._generate_url_list(league=league)
        league_matches_dfs = []

        for i, url in enumerate(url_list):
            try:
                matches_df = pd.read_csv(url)
                matches_df['Season'] = league.year_start + i
                league_matches_dfs.append(matches_df)
            except:
                break
        return league_matches_dfs[0] if len(league_matches_dfs) == 1 else pd.concat(league_matches_dfs)

    def _process_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        matches_df = matches_df[[
            'Date', 'Season', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'FTHG', 'FTAG', 'FTR'
        ]]
        matches_df = matches_df.rename(columns={
            'HomeTeam': 'Home Team',
            'AwayTeam': 'Away Team',
            'B365H': '1',
            'B365D': 'X',
            'B365A': '2',
            'FTHG': 'HG',
            'FTAG': 'AG',
            'FTR': 'Result'
        })
        return matches_df

    @staticmethod
    def _get_url_year_format(year: int):
        return f'{str(year)[2:]}{str(year + 1)[2:]}'

    def _generate_url_list(self, league: League):
        return [
            league.url.format(self._get_url_year_format(year=year))
            for year in range(league.year_start, date.today().year + 1)
        ]
