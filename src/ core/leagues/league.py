from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class League:
    """ League's entity class that stores all information about a league.

        Attributes
        ----------
        :param country: (str) league's country.
        :param name: (str) league's name.
        :param min_year: (int) league's data minimum year.
        :param category: (str) league's category (main/extra).
        :param url: (str) league's url from FootballData.co.uk website.
        :param fixture_url: (str) league's fixture url from FootyStats.com website.
    """

    country: str
    name: str
    min_year: int
    category: str
    url: str
    fixture_url: str


@dataclass(frozen=True, slots=True)
class LeagueConfig:
    """ League's config class that stores all user parameters about a created league.

        Attributes
        ----------
        :param league: (League) league instance.
        :param league_id: (str) unique identifier for the league.
        :param stats_columns: stats columns to be computed.
        :param start_year: (int) league's start year (season).
        :param match_history_window: (int) match history window to compute the league's statistics.
        :param goal_diff_margin: (int) minimum goal difference for marginal win/loss.
        :param odd_1_mask_range: ((int, int) | None) odd-1 range mask. Defaults to None.
        :param odd_x_mask_range: ((int, int) | None) odd-X range mask. Defaults to None.
        :param odd_2_mask_range: ((int, int) | None) odd-2 range mask. Defaults to None.
        :param aggregation: (str) rolling statistics aggregation method ("sum" or "mean"). Defaults to "sum".
    """

    league: League
    league_id: str
    start_year: int
    stats_columns: list[str]
    match_history_window: int = 3
    goal_diff_margin: int = 2
    aggregation: str = 'sum'
    odd_1_mask_range: tuple[float, float] | None = None
    odd_x_mask_range: tuple[float, float] | None = None
    odd_2_mask_range: tuple[float, float] | None = None
