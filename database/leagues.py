from datetime import date

_START_YEAR = 2015


class League:
    def __init__(self, name, base_url):
        self.name = name
        self.url_list = self._get_url_list(base_url)

    @staticmethod
    def _get_url_list(base_url):
        url_list = []

        for year in range(_START_YEAR, date.today().year):
            url_path = str(year)[2:] + str(year + 1)[2:]
            url_list.append(base_url.format(url_path))
        return url_list


class LeagueConfig:
    def __init__(self, config_dict):
        self._config_dict = config_dict

    def get_league_name(self):
        return self._config_dict['league_name']

    def get_directory_path(self):
        return self._config_dict['directory_path']

    def get_goal_diff_margin(self):
        return self._config_dict['goal_diff_margin']

    def get_last_n_matches(self):
        return self._config_dict['last_n_matches']

    def as_dict(self):
        return self._config_dict

    @staticmethod
    def generate_league_config(
            league_name,
            directory_path,
            goal_diff_margin,
            last_n_matches
    ):
        return LeagueConfig({
            'league_name': league_name,
            'directory_path': directory_path,
            'goal_diff_margin': goal_diff_margin,
            'last_n_matches': last_n_matches
        })


LEAGUES = {
    'England': League('England', 'https://www.football-data.co.uk/mmz4281/{}/E0.csv'),
    'Scotland': League('Scotland', 'https://www.football-data.co.uk/mmz4281/{}/SC0.csv'),
    'Germany': League('Germany', 'https://www.football-data.co.uk/mmz4281/{}/D1.csv'),
    'Italy': League('Italy', 'https://www.football-data.co.uk/mmz4281/{}/I1.csv'),
    'Spain': League('Spain', 'https://www.football-data.co.uk/mmz4281/{}/SP1.csv'),
    'France': League('France', 'https://www.football-data.co.uk/mmz4281/{}/F1.csv'),
    'Netherlands': League('Netherlands', 'https://www.football-data.co.uk/mmz4281/{}/N1.csv'),
    'Belgium': League('Belgium', 'https://www.football-data.co.uk/mmz4281/{}/B1.csv'),
    'Portugal': League('Portugal', 'https://www.football-data.co.uk/mmz4281/{}/P1.csv'),
    'Turkey': League('Turkey', 'https://www.football-data.co.uk/mmz4281/{}/T1.csv'),
    'Greece': League('Greece', 'https://www.football-data.co.uk/mmz4281/{}/G1.csv'),
}
