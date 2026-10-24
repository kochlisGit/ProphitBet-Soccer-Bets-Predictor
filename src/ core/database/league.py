from __future__ import annotations

import json
import os
import pickle
import shutil
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from src.core.leagues.league import League, LeagueConfig

if TYPE_CHECKING:
    import pandas as pd

    from src.core.network.downloaders.factory import DownloaderFactory


class LeagueDatabase:
    """ League database handler class.

        Functions:
            1. Loads all available league entries.
            2. Loads all stored leagues.
            3. Checks whether a league exists.
            4. Downloads league data.
            5. Stores the league data and config of a new league.
            6. Updates an existing league's data by id.
            7. Loads specified league by id.
            8. Deletes specified league by id.
    """

    def __init__(self):
        # Parsing JSON file and initializing all league entries.
        with open('storage/configs/leagues.json', mode='r', encoding='utf-8') as jsonfile:
            config = json.load(jsonfile)

        leagues = config.get('leagues', None)

        if not leagues:
            raise KeyError('"leagues.json" file is corrupted: "leagues" field is missing. Re-download project files.')

        # Initializing league instances.
        self._leagues = [League(**league) for league in leagues]

        # Initializing/Restoring database settings, storage directories and index.
        self._leagues_base_directory = config['leagues_base_directory']
        self._leagues_index_filepath = config['leagues_index_filepath']

        if not os.path.exists(path=self._leagues_base_directory):
            os.makedirs(name=self._leagues_base_directory, exist_ok=True)

        self._index = self._initialize_or_restore_index()
        self._downloader_factory: Optional[DownloaderFactory] = None

    @property
    def leagues(self) -> List[League]:
        return self._leagues

    @property
    def index(self) -> Dict[str, LeagueConfig]:
        return self._index

    def _initialize_or_restore_index(self) -> Dict[str, LeagueConfig]:
        """ Initializes an empty index (Dict) or restores the previously stored index, """

        if not os.path.exists(path=self._leagues_index_filepath):
            return {}
        else:
            with open(self._leagues_index_filepath, 'rb') as pklfile:
                index = pickle.load(pklfile)

            # Validate index directories.
            validated_index = {
                league_id: league
                for league_id, league in index.items()
                if os.path.exists(path=self._build_league_directory(league_id=league_id))
            }
            return validated_index

    def _build_league_directory(self, league_id: str) -> str:
        """ Builds the directory of a league using its id (user-defined name).

            Args
            ----
            :param league_id: (str) specified league's identifier.
        """

        return f'{self._leagues_base_directory}/{league_id}'

    def _update_index(self):
        """ Updates the database index and stores it in the pre-defined path. """

        with open(self._leagues_index_filepath, 'wb') as pklfile:
            pickle.dump(self._index, pklfile)

    def get_league_ids(self) -> list[str]:
        """ Gets a list of all league ids, sorted in ascending order. """

        return [] if len(self._index) == 0 else list(self._index.keys())

    def league_exists(self, league_id: str) -> bool:
        return league_id in self._index

    def download_league(self, league_config: LeagueConfig) -> Optional[pd.DataFrame]:
        """ Downloads the requested league, computes statistics and returns the resulting dataframe.

            Args
            ----
            :param league_config: (LeagueConfig) league's config provided by the user.
        """

        # Downloading dependencies are only needed for create/update actions,
        # not while restoring the lightweight league index at startup.
        from src.core.network.downloaders.factory import DownloaderFactory
        from src.core.preprocessing.statistics import StatisticsEngine

        if self._downloader_factory is None:
            self._downloader_factory = DownloaderFactory()

        # Download match data per season.
        downloader = self._downloader_factory.get_downloader(category=league_config.league.category)
        df = downloader.download(league_config=league_config)

        # Compute statistics per season.
        stats_engine = StatisticsEngine(
            match_history_window=league_config.match_history_window,
            goal_diff_margin=league_config.goal_diff_margin,
            aggregation=league_config.aggregation
        )
        df = stats_engine.compute_stats(df=df, stats_columns=league_config.stats_columns)

        # Sorting the league data by Date in descending order and by Home in ascending one.
        df = df.sort_values(by=['Date'], ascending=False, ignore_index=True)
        return df

    def save_league(self, df: pd.DataFrame, league_config: LeagueConfig):
        """ Stores the league into the database and updates the database  index.

            Args
            ----
            :param df: (DataFrame) the dataframe to be stored in the database.
            :param league_config: (LeagueConfig) league's config provided by the user.
        """

        # Store data.
        leagues_directory = f'{self._build_league_directory(league_id=league_config.league_id)}/data'
        os.makedirs(name=leagues_directory, exist_ok=True)
        df.to_feather(path=f'{leagues_directory}/dataset.feather')

        # Update index.
        self._index[league_config.league_id] = league_config
        self._update_index()

    def update_league(self, league_id: str) -> pd.DataFrame:
        """ Updates the league's data by id.

            Args
            ----
            :param league_id: (str) specified league's identifier.
        """

        import pandas as pd

        from src.core.preprocessing.filters import filter_league_odds

        # Loading existing league data.
        league_config = self._index[league_id]
        history_df = self.load_league(league_id=league_id)

        # Updating league's data.
        season = history_df.iloc[0]['Season']
        update_config = replace(league_config, start_year=season)
        update_df = self.download_league(league_config=update_config)

        if update_df is None:
            return history_df

        # Applying odd filters.
        update_df = filter_league_odds(
            df=update_df,
            odd_1_mask_range=league_config.odd_1_mask_range,
            odd_x_mask_range=league_config.odd_x_mask_range,
            odd_2_mask_range=league_config.odd_2_mask_range,
            validate_dataframe=False
        )
        if update_df is None:
            return history_df

        df = pd.concat((update_df, history_df[history_df['Season'] < season]), axis=0, ignore_index=True)
        self.save_league(df=df, league_config=league_config)
        return df

    def load_league(self, league_id: str) -> pd.DataFrame:
        """ Loads an existing league data by id.

            Args
            ----
            :param league_id: (str) specified league's identifier.
        """

        import pandas as pd

        leagues_directory = self._build_league_directory(league_id=league_id)
        league_dataset_filepath = f'{leagues_directory}/data/dataset.feather'
        return pd.read_feather(path=league_dataset_filepath)

    def delete_league(self, league_id: str):
        """ Deletes all league's downloaded data and removes its id from index.

            Args
            ----
            :param league_id: (str) specified league's identifier.
        """

        # Deleting league's data.
        self._index.pop(league_id)
        shutil.rmtree(self._build_league_directory(league_id=league_id), ignore_errors=True)

        # Updating index.
        self._update_index()

    def store_analysis(self, league_id: str, report_dict: Dict[str, Any]):
        """ Stores analysis report into the league directory.

            Attributes
            ----------
            league_id: (str) specified league's identifier.
            report_dict: Dict[str, Any] analysis report dict generated by the analyzer.
        """

        league_dir = self._build_league_directory(league_id=league_id)
        fp = f'{league_dir}/analysis.pkl'
        with open(fp, mode='wb') as pickle_file:
            pickle.dump(report_dict, pickle_file)

    def get_analysis(self, league_id: str) -> Optional[Dict[str, Any]]:
        """ Retrieves the analysis report from the league directory. Returns None if no analysis report exists.

            Attributes
            ----------
            league_id: (str) specified league's identifier.
        """

        league_dir = self._build_league_directory(league_id=league_id)
        fp = f'{league_dir}/analysis.pkl'

        if not os.path.exists(path=fp):
            return None

        with open(fp, mode='rb') as pickle_file:
            return pickle.load(pickle_file)
