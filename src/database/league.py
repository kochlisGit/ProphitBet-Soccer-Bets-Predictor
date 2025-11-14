import os
import json
import pickle
import shutil
import pandas as pd
from typing import Dict, List, Optional
from src.preprocessing.statistics import StatisticsEngine
from src.network.leagues.downloaders.main import MainLeagueDownloader
from src.network.leagues.downloaders.extra import ExtraLeagueDownloader
from src.network.leagues.league import League
from src.network.netutils import check_internet_connection


class LeagueDatabase:
    """ League database handler class. """

    def __init__(self):
        with open('storage/network/leagues.json', mode='r') as jsonfile:
            leagues_cfg = json.load(jsonfile)

        # Initialize all available leagues.
        self._leagues = [League(**cfg) for cfg in leagues_cfg['leagues']]
        self._leagues_directory = leagues_cfg['leagues_directory']
        self._leagues_index_filepath = leagues_cfg['leagues_index_filepath']

        # Create league's directory file if it does not exist.
        if not os.path.exists(path=self._leagues_directory):
            os.makedirs(name=self._leagues_directory, exist_ok=True)

        # Initialize or Restore index.
        self._index = self._initialize_or_restore_index()

        # Initialize league downloaders.
        self._downloaders = {'main': MainLeagueDownloader(), 'extra': ExtraLeagueDownloader()}

    @property
    def leagues(self) -> List[League]:
        return self._leagues

    @property
    def index(self) -> Dict[str, League]:
        return self._index

    def get_league_ids(self) -> list[str]:
        """ Gets a list of all league ids, sorted in ascending order. """

        return [] if len(self._index) == 0 else sorted(self._index.keys())

    def create_league(self, league: League) -> Optional[pd.DataFrame]:
        """ Downloads the league data and stores them into the database. """

        if not check_internet_connection():
            return None

        df = self._download_league(league=league, start_year=None)

        if df is not None:
            self.save_league(df=df, league=league)

        return df.reset_index(drop=True)

    def league_exists(self, league_id: str) -> bool:
        return league_id in self._index

    def update_league(self, league_id: str) -> Optional[pd.DataFrame]:
        league = self._index[league_id]
        history_df = self.load_league(league_id=league_id)
        season = history_df.iloc[0]['Season']
        update_df = self._download_league(league=league, start_year=season)

        if update_df is not None:
            df = pd.concat((update_df, history_df[history_df['Season'] < season]), axis=0, ignore_index=True)
            self.save_league(df=df, league=league)
        else:
            df = history_df

        return df.reset_index(drop=True)

    def save_league(self, df: pd.DataFrame, league: League):
        """ Stores the league into the database and updates the database  index. """

        # Store data.
        leagues_directory = f'{self._build_league_directory(league_id=league.league_id)}/data'
        os.makedirs(name=leagues_directory, exist_ok=True)
        df.to_csv(f'{leagues_directory}/dataset.csv', index=False)

        # Update index.
        self._index[league.league_id] = league
        self._save_index()

    def load_league(self, league_id: str) -> Optional[pd.DataFrame]:
        """ Loads a league using its id and returns the loaded data.
            If no such league exists, it updates the database index.
        """

        leagues_directory = self._build_league_directory(league_id=league_id)
        league_dataset_filepath = f'{leagues_directory}/data/dataset.csv'

        if os.path.exists(league_dataset_filepath):
            return pd.read_csv(league_dataset_filepath)
        else:
            self.delete_league(league_id=league_id)
            return None

    def delete_league(self, league_id: str):
        """ Deletes all league's downloaded data and removes its id from index. """

        self._index.pop(league_id)
        shutil.rmtree(self._build_league_directory(league_id=league_id), ignore_errors=True)
        self._save_index()

    def _initialize_or_restore_index(self) -> Dict[str, League]:
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

    def _save_index(self):
        """ Stores the index to the pre-defined index filepath. The index is stored as a pickle object. """

        with open(self._leagues_index_filepath, 'wb') as pklfile:
            pickle.dump(self._index, pklfile)

    def _build_league_directory(self, league_id: str) -> str:
        """ Builds the directory of a league using its id (user-defined name). """

        return f'{self._leagues_directory}/{league_id}'

    def _download_league(self, league: League, start_year: Optional[int] = None) -> Optional[pd.DataFrame]:
        """ Downloads the selected league data. """

        if start_year is None:
            start_year = league.start_year

        # Download match data per season.
        downloader = self._downloaders[league.category]
        df = downloader.download(league=league, start_year=start_year)

        if df is None:
            return None

        # Compute statistics per season.
        stats_engine = StatisticsEngine(
            match_history_window=league.match_history_window,
            goal_diff_margin=league.goal_diff_margin
        )
        df = stats_engine.compute_stats(df=df, stat_columns=league.stats_columns)

        # Filter odds.
        odd_1_filter = league.odd_1_range
        if odd_1_filter is not None:
            min_odd, max_odd = odd_1_filter
            mask = ((df['1'] >= min_odd) & (df['1'] <= max_odd))
            df = df[mask]
        odd_x_filter = league.odd_x_range
        if odd_x_filter is not None:
            min_odd, max_odd = odd_x_filter
            mask = ((df['X'] >= min_odd) & (df['X'] <= max_odd))
            df = df[mask]
        odd_2_filter = league.odd_2_range
        if odd_2_filter is not None:
            min_odd, max_odd = odd_2_filter
            mask = ((df['2'] >= min_odd) & (df['2'] <= max_odd))
            df = df[mask]

        return df
