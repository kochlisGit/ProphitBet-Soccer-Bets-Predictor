from database.leagues import LEAGUES, LeagueConfig
from database.statistics import StatisticsEngine
from urllib.request import urlretrieve
import os
import shutil
import csv


class LeagueRepository:
    def __init__(self):
        self._repository_path = 'database/leagues/'
        self._directory_path = self._repository_path + '{}/'
        self._downloads_filepath = self._directory_path + '{}'
        self._model_path = 'models/checkpoints/{}'
        self._config_filename = 'config.csv'

        self.basic_columns = [
            'Date', 'Home Team', 'Away Team', '1', 'X', '2', 'HG',  'AG', 'Result'
        ]
        self.stats_columns = [
            'HW', 'HL', 'HGF', 'HGD-W', 'HGD-L', 'HGD', 'HW%', 'HWD%',
            'AW', 'AL', 'AGF', 'AGD-W', 'AGD-L', 'AGD', 'AW%', 'AWD%'
        ]
        self.all_columns = self.basic_columns + self.stats_columns

        self._datasheet_column_map = {
            'Date': {'column_name': 'Date', 'column_type': lambda x: x},
            'Home Team': {'column_name': 'HomeTeam', 'column_type': lambda x: x},
            'Away Team': {'column_name': 'AwayTeam', 'column_type': lambda x: x},
            '1': {'column_name': 'B365H', 'column_type': lambda x: float(x)},
            'X': {'column_name': 'B365D', 'column_type': lambda x: float(x)},
            '2': {'column_name': 'B365A', 'column_type': lambda x: float(x)},
            'HG': {'column_name': 'FTHG', 'column_type': lambda x: int(x)},
            'AG': {'column_name': 'FTAG', 'column_type': lambda x: int(x)},
            'Result': {'column_name': 'FTR', 'column_type': lambda x: x}
        }

        self._stats_engine = StatisticsEngine()
        self._stats_operations = [
            self._stats_engine.compute_last_wins,
            self._stats_engine.compute_last_losses,
            self._stats_engine.compute_last_goal_forward,
            self._stats_engine.compute_last_goal_diff_wins,
            self._stats_engine.compute_last_goal_diff_losses,
            self._stats_engine.compute_last_goal_diffs,
            self._stats_engine.compute_total_win_rate,
            self._stats_engine.compute_total_win_draw_rate
        ]

        self._initialize_directory(self._repository_path)

    @staticmethod
    def _initialize_directory(directory_path):
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)

    def _load_config(self, directory_path):
        config_filepath = self._downloads_filepath.format(directory_path, self._config_filename)

        with open(config_filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            return LeagueConfig(next(reader))

    def _store_config(self, config):
        config_filepath = self._downloads_filepath.format(config.get_directory_path(), self._config_filename)
        config_dict = config.as_dict()

        with open(config_filepath, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, config_dict.keys())
            writer.writeheader()
            writer.writerow(config_dict)

    def model_exists(self, directory_path):
        return os.path.exists(self._model_path.format(directory_path))

    def download_league(self, config):
        league = LEAGUES[config.get_league_name()]
        directory_path = config.get_directory_path()
        self._initialize_directory(self._directory_path.format(directory_path))

        for i, url in enumerate(league.url_list):
            filename = self._downloads_filepath.format(directory_path, i) + '.csv'
            urlretrieve(url, filename)
        self._store_config(config)

    def update_league(self, directory_path):
        config = self._load_config(directory_path)
        league = LEAGUES[config.get_league_name()]
        url_list = league.url_list
        update_url = url_list[-1]
        filename = str(len(url_list) - 1) + '.csv'
        urlretrieve(update_url, self._downloads_filepath.format(directory_path, filename))

    @staticmethod
    def get_all_leagues():
        return LEAGUES

    def get_downloaded_leagues(self):
        return os.listdir(self._repository_path)

    def delete_league(self, directory_path):
        shutil.rmtree(self._directory_path.format(directory_path))

        model_directory_path = self._model_path.format(directory_path)
        if os.path.exists(model_directory_path):
            shutil.rmtree(model_directory_path)

    def read_league_results_and_stats(self, directory_path):
        results_and_stats = []

        config = self._load_config(directory_path)
        self._stats_engine.set_options(
            home_column=self.basic_columns.index('Home Team'),
            away_column=self.basic_columns.index('Away Team'),
            home_goals_column=self.basic_columns.index('HG'),
            away_goals_column=self.basic_columns.index('AG'),
            result_column=self.basic_columns.index('Result'),
            last_n=int(config.get_last_n_matches()),
            goal_diffs_margin=int(config.get_goal_diff_margin())
        )

        downloaded_files = os.listdir(self._directory_path.format(directory_path))
        downloaded_files.remove('config.csv')
        downloaded_files.reverse()

        for filename in downloaded_files:
            filepath = self._downloads_filepath.format(directory_path, filename)

            with open(filepath, 'r', encoding='unicode_escape') as csvfile:
                reader = csv.reader(csvfile)
                all_datasheet_columns = next(reader)
                column_indices = {
                    name: all_datasheet_columns.index(mapping['column_name'])
                    for name, mapping in self._datasheet_column_map.items()
                }
                league_results = []
                for row in reader:
                    try:
                        league_results.append([
                            self._datasheet_column_map[col_name]['column_type'](row[column_indices[col_name]])
                            for col_name in self.basic_columns
                        ])
                    except:
                        continue

                # league_results = [[
                #         self._datasheet_column_map[col_name]['column_type'](row[column_indices[col_name]])
                #         for col_name in self.basic_columns
                #     ] for row in reader]
            league_results.reverse()
            league_stats = self._compute_stats(
                league_results,
                self._stats_engine.home_column,
                self._stats_engine.away_column
            )
            merged_data = self._merge_results_and_stats(league_results, league_stats)
            results_and_stats += merged_data
        return results_and_stats

    def _compute_stats(self, league_results, home_team_column, away_team_column):
        stats = []

        for i, result in enumerate(league_results):
            previous_results = league_results[i+1:]
            home_stats = [
                oper(previous_matches=previous_results, team_name=result[home_team_column], is_home=True)
                for oper in self._stats_operations
            ]

            if None in home_stats:
                continue

            away_stats = [
                oper(previous_matches=previous_results, team_name=result[away_team_column], is_home=False)
                for oper in self._stats_operations
            ]

            if None in away_stats:
                continue

            stats.append(home_stats + away_stats)
        return stats

    @staticmethod
    def _merge_results_and_stats(league_results, stats):
        merged_results = []

        for i, statistic_results in enumerate(stats):
            merged_results.append(league_results[i] + statistic_results)
        return merged_results
