import csv
import json
import os
import pandas as pd
from database.entities.league import League
from database.network.netutils import check_internet_connection
from database.network.footballdata.extra import ExtraLeagueAPI
from database.network.footballdata.main import MainLeagueAPI
from preprocessing.statistics import StatisticsEngine


class LeagueRepository:
    def __init__(self, available_leagues_filepath: str, saved_leagues_directory: str):
        self._available_leagues_filepath = available_leagues_filepath
        self._saved_leagues_directory = saved_leagues_directory

        os.makedirs(self._saved_leagues_directory, exist_ok=True)

    def get_all_available_leagues(self) -> dict:
        with open(file=self._available_leagues_filepath, mode='r', encoding='utf=8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)

            return {(row[0], row[1]): League(
                country=row[0],
                name=row[1],
                url=row[2],
                year_start=int(row[3]),
                league_type=row[4],
                fixtures_url=row[5]
            ) for row in reader}

    def get_all_saved_leagues(self) -> list:
        all_filepaths = os.listdir(self._saved_leagues_directory)
        return list({os.path.basename(filepath).split(sep='.')[0] for filepath in all_filepaths})

    @staticmethod
    def get_all_available_columns() -> list:
        return StatisticsEngine.Columns

    def league_exists(self, league_name: str) -> bool:
        league_filepath = f'{self._saved_leagues_directory}{league_name}.csv'
        return os.path.exists(league_filepath)

    def _store_league(
            self,
            matches_df: pd.DataFrame,
            league_config: dict,
            league_name: str,
            store_league_config: bool
    ) -> pd.DataFrame:
        matches_df = StatisticsEngine(
            matches_df=matches_df,
            last_n_matches=league_config['last_n_matches'],
            goal_diff_margin=league_config['goal_diff_margin']
        ).compute_statistics(statistic_columns=league_config['statistic_columns'])

        league_filepath = f'{self._saved_leagues_directory}{league_name}.csv'
        matches_df.to_csv(league_filepath, index=False)

        if store_league_config:
            league_config_filepath = f'{self._saved_leagues_directory}{league_name}.json'
            with open(file=league_config_filepath, mode='w', encoding='utf-8') as fp:
                json.dump(league_config, fp)
        return matches_df

    def create_league(
            self,
            league: League,
            last_n_matches: int,
            goal_diff_margin: int,
            statistic_columns: list,
            league_name: str
    ) -> (pd.DataFrame, League) or (None, None):
        if check_internet_connection() and not self.league_exists(league_name=league_name):
            if league.league_type == 'main':
                matches_df = MainLeagueAPI().download(league=league)
            elif league.league_type == 'extra':
                matches_df = ExtraLeagueAPI().download(league=league)
            else:
                raise NotImplementedError(f'League_type = {league.league_type} has not been implemented')

            league_config = {
                'country': league.country,
                'name': league.name,
                'last_n_matches': last_n_matches,
                'goal_diff_margin': goal_diff_margin,
                'statistic_columns': statistic_columns
            }
            return self._store_league(
                matches_df=matches_df,
                league_config=league_config,
                league_name=league_name,
                store_league_config=True
            ), league
        else:
            return None, None

    def update_league(self, league_name: str) -> (pd.DataFrame, League) or (None, None):
        if check_internet_connection() and self.league_exists(league_name=league_name):
            config_filepath = f'{self._saved_leagues_directory}{league_name}.json'
            with open(config_filepath, 'r', encoding='utf-8') as fp:
                league_config = json.load(fp)

            if self.delete_league(league_name=league_name):
                league_key = (league_config['country'], league_config['name'])
                league = self.get_all_available_leagues()[league_key]
                return self.create_league(
                    league=league,
                    last_n_matches=league_config['last_n_matches'],
                    goal_diff_margin=league_config['goal_diff_margin'],
                    statistic_columns=league_config['statistic_columns'],
                    league_name=league_name
                )
            else:
                return None, None
        else:
            return None, None

    def load_league(self, league_name: str) -> (pd.DataFrame, League) or (None, None):
        if self.league_exists(league_name=league_name):
            matches_df = pd.read_csv(f'{self._saved_leagues_directory}{league_name}.csv')

            with open(f'{self._saved_leagues_directory}{league_name}.json', 'r', encoding='utf-8') as fp:
                league_config = json.load(fp)
            league = self.get_all_available_leagues()[(league_config['country'], league_config['name'])]
            return matches_df, league
        else:
            return None, None

    def delete_league(self, league_name: str) -> bool:
        league_filepath = f'{self._saved_leagues_directory}{league_name}.csv'
        league_config_filepath = f'{self._saved_leagues_directory}{league_name}.json'

        if os.path.exists(league_filepath):
            os.remove(league_filepath)
            os.remove(league_config_filepath)
            return True
        else:
            return False
