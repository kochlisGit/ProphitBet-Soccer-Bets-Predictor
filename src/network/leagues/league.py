from typing import Any, Dict, List, Optional, Tuple


class League:
    """ League entry class, which is used as a placeholder for the league's attributes. """

    def __init__(
            self,
            country: str,
            name: str,
            start_year: int,
            category: str,
            url: str,
            fixture: str,
            league_id: Optional[str] = None,
            match_history_window: Optional[int] = None,
            goal_diff_margin: Optional[int] = None,
            stats_columns: Optional[List[str]] = None,
            odd_1_range: Optional[Tuple[float, float]] = None,
            odd_x_range: Optional[Tuple[float, float]] = None,
            odd_2_range: Optional[Tuple[float, float]] = None
    ):
        """
            :param country: The league's country.
            :param name: The league's name.
            :param start_year: The league's starting year.
            :param category: The league's category.
            :param url: The football-data url, which is used to download the csv league files.
            :param fixture: The footystats url, which is used to parse the league's fixtures.
            :param league_id: The league's id (defined by the user).
            :param match_history_window: The statistics match history window (defined by the user).
            :param goal_diff_margin: The statistics goal diff margin (defined by the user).
            :param stats_columns: The selected statistics columns (defined by the user).
            :param odd_1_range: The selected odd-1 download filter (defined by the user).
            :param odd_x_range: The selected odd-x download filter (defined by the user).
            :param odd_2_range: The selected odd-2 download filter (defined by the user).
        """

        self._country = country
        self._name = name
        self._start_year = start_year
        self._category = category
        self._url = url
        self._fixture = fixture

        self._league_id = league_id
        self._match_history_window = match_history_window
        self._goal_diff_margin = goal_diff_margin
        self._stats_columns = stats_columns
        self._odd_1_range = odd_1_range
        self._odd_x_range = odd_x_range
        self._odd_2_range = odd_2_range

    def __str__(self):
        return f'{self._country}-{self._name}' if self._league_id is None else self._league_id

    @property
    def country(self) -> str:
        return self._country

    @property
    def name(self) -> str:
        return self._name

    @property
    def start_year(self) -> int:
        return self._start_year

    @property
    def category(self) -> str:
        return self._category

    @property
    def url(self):
        return self._url

    @property
    def fixture(self):
        return self._fixture

    @property
    def league_id(self) -> Optional[str]:
        return self._league_id

    @property
    def match_history_window(self) -> Optional[int]:
        return self._match_history_window

    @property
    def goal_diff_margin(self) -> Optional[int]:
        return self._goal_diff_margin

    @property
    def stats_columns(self) -> Optional[List[str]]:
        return self._stats_columns

    @property
    def odd_1_range(self) -> Optional[Tuple[float, float]]:
        return self._odd_1_range

    @property
    def odd_x_range(self) -> Optional[Tuple[float, float]]:
        return self._odd_x_range

    @property
    def odd_2_range(self) -> Optional[Tuple[float, float]]:
        return self._odd_2_range

    def clone(
            self,
            start_year: int,
            league_id: str,
            match_history_window: int,
            goal_diff_margin: int,
            stats_columns: Optional[List[str]],
            odd_1_range: Optional[Tuple[float, float]] = None,
            odd_x_range: Optional[Tuple[float, float]] = None,
            odd_2_range: Optional[Tuple[float, float]] = None
    ):
        """ Closes the current league instance and modifies the league_id and start year only. """

        if start_year < self._start_year:
            start_year = self._start_year

        return League(
            country=self._country,
            name=self._name,
            start_year=start_year,
            category=self._category,
            url=self._url,
            fixture=self._fixture,
            league_id=league_id,
            match_history_window=match_history_window,
            goal_diff_margin=goal_diff_margin,
            stats_columns=stats_columns,
            odd_1_range=odd_1_range,
            odd_x_range=odd_x_range,
            odd_2_range=odd_2_range
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'country': self._country,
            'name': self._name,
            'start_year': self._start_year,
            'category': self._category,
            'url': self._url,
            'fixture': self._fixture,
            'league_id': self._league_id,
            'match_history_window': self._match_history_window,
            'goal_diff_margin': self._goal_diff_margin,
            'stats_columns': self._stats_columns,
            'odd_1_range': self._odd_1_range,
            'odd_x_range': self._odd_x_range,
            'odd_2_range': self._odd_2_range
        }
