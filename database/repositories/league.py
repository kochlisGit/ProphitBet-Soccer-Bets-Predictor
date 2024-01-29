import os
import pickle
import shutil
import pandas as pd
from database.entities.leagues.league import League, LeagueConfig
from database.network.downloaders.extra import ExtraLeagueDownloader
from database.network.downloaders.main import MainLeagueDownloader
from preprocessing.statistics import StatisticsEngine


class LeagueRepository:
    def __init__(
            self,
            leagues_directory: str,
            leagues_index_filepath: str,
            all_leagues_dict: dict[str, list]
    ):
        self._leagues_directory = leagues_directory
        self._leagues_index_filepath = leagues_index_filepath
        self._all_leagues_dict = all_leagues_dict

        os.makedirs(name=leagues_directory, exist_ok=True)

        if not os.path.exists(leagues_index_filepath):
            self._index = {}
        else:
            self._load_index()

    @property
    def all_leagues_dict(self) -> dict[str, list[League]]:
        return self._all_leagues_dict

    @property
    def index(self) -> dict[str, LeagueConfig]:
        return self._index

    def _save_index(self):
        with open(self._leagues_index_filepath, 'wb') as pklfile:
            pickle.dump(self._index, pklfile)

    def _load_index(self):
        with open(self._leagues_index_filepath, 'rb') as pklfile:
            self._index = pickle.load(pklfile)

    def _get_league_directory(self, league_id: str) -> str:
        return f'{self._leagues_directory}/{league_id}'

    def get_created_leagues(self) -> list[str]:
        return [] if len(self._index) == 0 else sorted(self._index.keys())

    def get_league_config(self, league_id: str) -> LeagueConfig:
        return self._index[league_id]

    def _download_league_data(self, league_config: LeagueConfig, year_start: int):
        league = league_config.league

        if league.category == 'main':
            df = MainLeagueDownloader().download(league=league, year_start=year_start)
        elif league.category == 'extra':
            df = ExtraLeagueDownloader().download(league=league, year_start=year_start)
        else:
            raise NotImplementedError(f'Not implemented league category: "{league.category}"')

        if df is not None:
            stats_engine = StatisticsEngine(
                match_history_window=league_config.match_history_window,
                goal_diff_margin=league_config.goal_diff_margin
            )
            df = stats_engine.compute_statistics(matches_df=df, features=league_config.features)

        columns_to_drop = [col for col in ['1', 'X', '2'] if col not in league_config.features]
        if len(columns_to_drop) > 0:
            df.drop(columns=columns_to_drop, inplace=True)
        return df

    def create_league(self, league_config: LeagueConfig) -> pd.DataFrame or None:
        df = self._download_league_data(league_config=league_config, year_start=league_config.league.year_start)

        if df is not None:
            self.save_league(df=df, league_config=league_config)

        return df

    def update_league(self, league_id: str) -> pd.DataFrame or None:
        league_config = self._index[league_id]
        history_df = self.load_league(league_id=league_id)
        last_season = history_df.iloc[-1]['Season']
        update_df = self._download_league_data(league_config=league_config, year_start=last_season)

        if update_df is not None:
            df = pd.concat((update_df, history_df[history_df['Season'] < last_season]), axis=0, ignore_index=True)
            self.save_league(df=df, league_config=league_config)
        else:
            df = history_df

        return df

    def save_league(self, df: pd.DataFrame, league_config: LeagueConfig):
        leagues_directory = self._get_league_directory(league_id=league_config.league_id)
        os.makedirs(name=leagues_directory, exist_ok=True)
        df.to_csv(f'{leagues_directory}/dataset.csv', index=False)

        self._index[league_config.league_id] = league_config
        self._save_index()

    def load_league(self, league_id: str) -> pd.DataFrame or None:
        leagues_directory = self._get_league_directory(league_id=league_id)
        league_dataset_filepath = f'{leagues_directory}/dataset.csv'

        if os.path.exists(league_dataset_filepath):
            return pd.read_csv(league_dataset_filepath)
        else:
            self.delete_league(league_id=league_id)
            return None

    def delete_league(self, league_id: str):
        shutil.rmtree(self._get_league_directory(league_id=league_id), ignore_errors=True)
        del self._index[league_id]

        self._save_index()
