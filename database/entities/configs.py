from database.entities.leagues import League


class LeagueConfig:
    def __init__(
            self,
            league: League,
            league_identifier: str,
            last_n_matches: int,
            goal_diff_margin: int
    ):
        self._league = league
        self._league_identifier = league_identifier
        self._last_n_matches = last_n_matches
        self._goal_diff_margin = goal_diff_margin

    @property
    def league(self) -> League:
        return self._league

    @property
    def league_identifier(self) -> str:
        return self._league_identifier

    @property
    def last_n_matches(self) -> int:
        return self._last_n_matches

    @property
    def goal_diff_margin(self) -> int:
        return self._goal_diff_margin
