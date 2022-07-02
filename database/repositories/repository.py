from database.network.footballdata.api import FootballDataLeaguesAPI
from database.preprocessing.stats import LeagueStats
from database.entities.configs import LeagueConfig
from database.entities.leagues import League
from models.neuralnet.nn import FCNet
from models.randomforest.rf import RandomForest
import pandas as pd
import pickle
import os
import shutil


class LeagueRepository:
    def __init__(self, repository_directory: str, checkpoint_directory: str):
        self._repository_directory = repository_directory
        self._checkpoint_directory = checkpoint_directory

    @property
    def repository_directory(self):
        return self._repository_directory

    @property
    def checkpoint_directory(self):
        return self._checkpoint_directory

    @property
    def football_data_api(self):
        return FootballDataLeaguesAPI()

    @property
    def config_name(self):
        return 'config.cfg'

    @property
    def basic_columns(self) -> list:
        return [
            'Date', 'Home Team', 'Away Team', '1', 'X', '2', 'HG',  'AG', 'Result'
        ]

    @property
    def basic_column_indices(self) -> dict:
        return {
            col_name: self.basic_columns.index(col_name) for col_name in self.basic_columns
        }

    @property
    def stats_columns(self) -> list:
        return [
            'HW', 'HL', 'HGF', 'HGD-W', 'HGD-L', 'HW%', 'HD%',
            'AW', 'AL', 'AGF', 'AGD-W', 'AGD-L', 'AW%', 'AD%'
        ]

    @property
    def all_columns(self) -> list:
        return self.basic_columns + self.stats_columns

    @staticmethod
    def _initialize_directory(directory_path: str):
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)

    def get_downloaded_league_configs(self) -> dict:
        downloaded_leagues = {}
        league_dirs = os.listdir(self.repository_directory)

        for league_dir in league_dirs:
            config_filepath = f'{self.repository_directory}/{league_dir}/{self.config_name}'

            with open(config_filepath, 'rb') as config_file:
                downloaded_leagues[league_dir] = pickle.load(config_file)
        return downloaded_leagues

    def get_saved_models(self, league_identifier: str) -> dict:
        models = {}

        league_checkpoint_dir = f'{self.checkpoint_directory}/{league_identifier}'
        if os.path.exists(league_checkpoint_dir):
            checkpoints = os.listdir(league_checkpoint_dir)
    
            for checkpoint_name in checkpoints:
                if 'nn' in checkpoint_name:
                    fcnet = FCNet(
                        input_shape=(),
                        checkpoint_path=self.checkpoint_directory,
                        league_identifier=league_identifier,
                        model_name=league_identifier
                    )
                    fcnet.load()
                    models[checkpoint_name] = fcnet
                elif 'rf' in checkpoint_name:
                    rf = RandomForest(
                        input_shape=(),
                        checkpoint_path=self.checkpoint_directory,
                        league_identifier=league_identifier,
                        model_name=league_identifier,
                        calibrate_model=False
                    )
                    rf.load()
                    models[checkpoint_name] = rf
        return models

    def delete_league(self, directory_path):
        shutil.rmtree(f'{self.repository_directory}/{directory_path}')

        checkpoint_path = f'{self.checkpoint_directory}/{directory_path}'
        if os.path.exists(checkpoint_path):
            shutil.rmtree(f'checkpoint_path')

    def _store_config(self, league_config: LeagueConfig):
        config_filepath = f'{self.repository_directory}/{league_config.league_identifier}/{self.config_name}'

        with open(config_filepath, 'wb') as config_file:
            pickle.dump(league_config, config_file)

    def download_repository(self, league: League, league_config: LeagueConfig) -> bool:
        league_directory = f'{self.repository_directory}/{league_config.league_identifier}'
        self._initialize_directory(directory_path=league_directory)

        download_result = self.football_data_api.download_league(
            league=league,
            download_directory=league_directory
        )

        if download_result:
            self._store_config(league_config=league_config)
        return download_result

    def update_repository(self, league_config: LeagueConfig):
        league = league_config.league
        league_directory = f'{self.repository_directory}/{league_config.league_identifier}'

        self.football_data_api.update_league(
            league=league,
            download_directory=league_directory
        )

    def compute_results_and_stats(self, league_config: LeagueConfig) -> pd.DataFrame:
        results_and_stats = []
        league_directory = f'{self.repository_directory}/{league_config.league_identifier}'
        downloaded_csv_files = sorted(os.listdir(league_directory), reverse=True)
        downloaded_csv_files.remove(self.config_name)
        column_indices = self.basic_column_indices

        for csv_name in downloaded_csv_files:
            csv_filepath = f'{self.repository_directory}/{league_config.league_identifier}/{csv_name}'
            match_history_df = pd.read_csv(csv_filepath)

            match_history = match_history_df[self.basic_columns].values.tolist()
            n_matches = len(match_history)

            stats_engine = LeagueStats(
                home_index=column_indices['Home Team'],
                away_index=column_indices['Away Team'],
                hg_index=column_indices['HG'],
                ag_index=column_indices['AG'],
                result_index=column_indices['Result'],
                last_n_matches=league_config.last_n_matches,
                goal_diff_margin=league_config.goal_diff_margin
            )

            for i, match in enumerate(match_history):
                previous_match_history = match_history[i+1:]
                if i == n_matches - 1:
                    break

                hw = stats_engine.compute_last_wins(
                    match_history=previous_match_history,
                    team_name=match_history[i][column_indices['Home Team']],
                    is_home=True
                )
                if hw is None:
                    continue

                hl = stats_engine.compute_last_losses(
                    match_history=previous_match_history,
                    team_name=match_history[i][column_indices['Home Team']],
                    is_home=True
                )
                if hl is None:
                    continue

                hgf = stats_engine.compute_last_goal_forward(
                    match_history=previous_match_history,
                    team_name=match_history[i][column_indices['Home Team']],
                    is_home=True
                )
                if hgf is None:
                    continue

                hgdw = stats_engine.compute_last_n_goal_diff_wins(
                    match_history=previous_match_history,
                    team_name=match_history[i][column_indices['Home Team']],
                    is_home=True
                )
                if hgdw is None:
                    continue

                hgdl = stats_engine.compute_last_n_goal_diff_losses(
                    match_history=previous_match_history,
                    team_name=match_history[i][column_indices['Home Team']],
                    is_home=True
                )
                if hgdl is None:
                    continue

                hw_perc = stats_engine.compute_total_win_rate(
                    match_history=previous_match_history,
                    team_name=match_history[i][column_indices['Home Team']],
                    is_home=True
                )
                if hw_perc is None:
                    continue

                hwd_perc = stats_engine.compute_total_draw_rate(
                    match_history=previous_match_history,
                    team_name=match_history[i][column_indices['Home Team']],
                    is_home=True
                )
                if hwd_perc is None:
                    continue

                aw = stats_engine.compute_last_wins(
                    match_history=previous_match_history,
                    team_name=match_history[i][column_indices['Away Team']],
                    is_home=False
                )
                if aw is None:
                    continue

                al = stats_engine.compute_last_losses(
                    match_history=previous_match_history,
                    team_name=match_history[i][column_indices['Away Team']],
                    is_home=False
                )
                if al is None:
                    continue

                agf = stats_engine.compute_last_goal_forward(
                    match_history=previous_match_history,
                    team_name=match_history[i][column_indices['Away Team']],
                    is_home=False
                )
                if agf is None:
                    continue

                agdw = stats_engine.compute_last_n_goal_diff_wins(
                    match_history=previous_match_history,
                    team_name=match_history[i][column_indices['Away Team']],
                    is_home=False
                )
                if agdw is None:
                    continue

                agdl = stats_engine.compute_last_n_goal_diff_losses(
                    match_history=previous_match_history,
                    team_name=match_history[i][column_indices['Away Team']],
                    is_home=False
                )
                if agdl is None:
                    continue

                aw_perc = stats_engine.compute_total_win_rate(
                    match_history=previous_match_history,
                    team_name=match_history[i][column_indices['Away Team']],
                    is_home=False
                )
                if aw_perc is None:
                    continue

                awd_perc = stats_engine.compute_total_draw_rate(
                    match_history=previous_match_history,
                    team_name=match_history[i][column_indices['Away Team']],
                    is_home=False
                )
                if awd_perc is None:
                    continue

                results_and_stats.append(
                    [match_history[i][column_indices[col_name]] for col_name in self.basic_columns] +
                    [hw, hl, hgf, hgdw, hgdl, hw_perc, hwd_perc, aw, al, agf, agdw, agdl, aw_perc, awd_perc]
                )
        return pd.DataFrame(data=results_and_stats, columns=self.all_columns)
