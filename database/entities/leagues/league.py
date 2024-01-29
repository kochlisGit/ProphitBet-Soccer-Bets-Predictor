from abc import ABC


class League(ABC):
    def __init__(self, country: str, name: str, year_start: int, category: str, data_url: str, fixtures_url: str):
        assert category == 'main' or category == 'extra', f'Not supported category: {category}'

        self._country = country
        self._name = name
        self._category = category
        self._data_url = data_url
        self._fixtures_url = fixtures_url

        self.year_start = year_start

    @property
    def country(self) -> str:
        return self._country

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> str:
        return self._category

    @property
    def data_url(self) -> str:
        return self._data_url

    @property
    def fixtures_url(self) -> str:
        return self._fixtures_url


class LeagueConfig:
    def __init__(
            self,
            league_id: str,
            league: League,
            match_history_window: int,
            goal_diff_margin: int,
            features: list[str]
    ):
        self._league_id = league_id
        self._league = league
        self._match_history_window = match_history_window
        self._goal_diff_margin = goal_diff_margin
        self._features = features

    @property
    def league_id(self) -> str:
        return self._league_id

    @property
    def league(self) -> League:
        return self._league

    @property
    def match_history_window(self) -> int:
        return self._match_history_window

    @property
    def goal_diff_margin(self) -> int:
        return self._goal_diff_margin

    @property
    def features(self) -> list[str]:
        return self._features
