from database.repository import LeagueRepository


class DatabaseManager:
    def __init__(self):
        self._league_repository = LeagueRepository()

    def download_league(self, config):
        self._league_repository.download_league(config)

    def update_league(self, directory_path):
        self._league_repository.update_league(directory_path)

    def get_downloaded_leagues(self):
        return self._league_repository.get_downloaded_leagues()

    def get_all_league_names(self):
        return self._league_repository.get_all_leagues().keys()

    def delete_league(self, directory_path):
        return self._league_repository.delete_league(directory_path)

    def read_league(self, directory_path):
        return self._league_repository.read_league_results_and_stats(directory_path)

    def read_league_headers(self):
        return self._league_repository.basic_columns + self._league_repository.stats_columns
