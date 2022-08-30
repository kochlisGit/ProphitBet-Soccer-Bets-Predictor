class LeagueStats:
    def __init__(
            self,
            home_index: int,
            away_index: int,
            hg_index: int,
            ag_index: int,
            result_index: int,
            last_n_matches: int,
            goal_diff_margin: int
    ):
        self._home_index = home_index
        self._away_index = away_index
        self._hg_index = hg_index
        self._ag_index = ag_index
        self._result_index = result_index
        self._last_n_matches = last_n_matches
        self._goal_diff_margin = goal_diff_margin

    @property
    def home_index(self) -> int:
        return self._home_index

    @property
    def away_index(self) -> int:
        return self._away_index

    @property
    def hg_index(self) -> int:
        return self._hg_index

    @property
    def ag_index(self) -> int:
        return self._ag_index

    @property
    def result_index(self) -> int:
        return self._result_index

    @property
    def last_n_matches(self) -> int:
        return self._last_n_matches

    @property
    def goal_diff_margin(self) -> int:
        return self._goal_diff_margin

    def compute_last_wins(self, match_history: list, team_name: str, is_home: bool) -> int or None:
        team_index = self.home_index if is_home else self.away_index
        last_n = self._last_n_matches
        last_wins = 0

        for match in match_history:
            if match[team_index] == team_name:
                result = match[self.result_index]

                if (is_home and result == 'H') or (not is_home and result == 'A'):
                    last_wins += 1

                last_n -= 1
                if last_n == 0:
                    break
        return None if last_n > 0 else last_wins

    def compute_last_losses(self, match_history: list, team_name: str, is_home: bool) -> int or None:
        team_index = self.home_index if is_home else self.away_index
        last_n = self._last_n_matches
        last_losses = 0

        for match in match_history:
            if match[team_index] == team_name:
                result = match[self.result_index]

                if (is_home and result == 'A') or (not is_home and result == 'H'):
                    last_losses += 1

                last_n -= 1
                if last_n == 0:
                    break
        return None if last_n > 0 else last_losses

    def compute_last_goal_forward(self, match_history: list, team_name: str, is_home: bool) -> int or None:
        team_index = self.home_index if is_home else self.away_index
        last_n = self._last_n_matches
        last_goals_forward = 0

        for match in match_history:
            if match[team_index] == team_name:
                last_goals_forward += match[self.hg_index] if is_home else match[self._ag_index]

                last_n -= 1
                if last_n == 0:
                    break
        return None if last_n > 0 else last_goals_forward

    def compute_last_goal_against(self, match_history: list, team_name: str, is_home: bool) -> int or None:
        team_index = self.home_index if is_home else self.away_index
        last_n = self._last_n_matches
        last_goals_against = 0

        for match in match_history:
            if match[team_index] == team_name:
                last_goals_against += match[self._ag_index] if is_home else match[self.hg_index]

                last_n -= 1
                if last_n == 0:
                    break
        return None if last_n > 0 else last_goals_against

    def compute_last_n_goal_diff_wins(self, match_history: list, team_name: str, is_home: bool) -> int or None:
        team_index = self.home_index if is_home else self.away_index
        last_n = self._last_n_matches
        last_n_goal_diff_wins = 0

        for match in match_history:
            if match[team_index] == team_name:
                home_goals = match[self.hg_index]
                away_goals = match[self.ag_index]

                if is_home:
                    if home_goals >= away_goals + self.goal_diff_margin:
                        last_n_goal_diff_wins += 1
                else:
                    if away_goals >= home_goals + self.goal_diff_margin:
                        last_n_goal_diff_wins += 1

                last_n -= 1
                if last_n == 0:
                    break
        return None if last_n > 0 else last_n_goal_diff_wins

    def compute_last_n_goal_diff_losses(self, match_history: list, team_name: str, is_home: bool) -> int or None:
        team_index = self.home_index if is_home else self.away_index
        last_n = self._last_n_matches
        last_n_goal_diff_losses = 0

        for match in match_history:
            if match[team_index] == team_name:
                home_goals = match[self.hg_index]
                away_goals = match[self.ag_index]

                if is_home:
                    if away_goals >= home_goals + self.goal_diff_margin:
                        last_n_goal_diff_losses += 1
                else:
                    if home_goals >= away_goals + self.goal_diff_margin:
                        last_n_goal_diff_losses += 1

                last_n -= 1
                if last_n == 0:
                    break
        return None if last_n > 0 else last_n_goal_diff_losses

    def compute_last_goal_diffs(self, match_history: list, team_name: str, is_home: bool) -> int or None:
        team_index = self.home_index if is_home else self.away_index
        last_n = self._last_n_matches
        goals_scored = 0
        goals_received = 0

        for match in match_history:
            if match[team_index] == team_name:
                home_goals = match[self.hg_index]
                away_goals = match[self.ag_index]

                if is_home:
                    goals_scored += home_goals
                    goals_received += away_goals
                else:
                    goals_scored += away_goals
                    goals_received += home_goals

                last_n -= 1
                if last_n == 0:
                    break
        return None if last_n > 0 else goals_scored - goals_received

    def compute_total_win_rate(self, match_history: list, team_name: str, is_home: bool) -> float or None:
        team_index = self.home_index if is_home else self.away_index
        last_played = 0
        last_wins = 0

        for match in match_history:
            if match[team_index] == team_name:
                last_played += 1
                result = match[self.result_index]

                if (is_home and result == 'H') or (not is_home and result == 'A'):
                    last_wins += 1
        return None if last_played == 0 else last_wins/last_played

    def compute_total_draw_rate(self, match_history: list, team_name: str, is_home: bool) -> float or None:
        team_index = self.home_index if is_home else self.away_index
        last_played = 0
        last_draws = 0

        for match in match_history:
            if match[team_index] == team_name:
                last_played += 1
                result = match[self.result_index]

                if result == 'D':
                    last_draws += 1
        return None if last_played == 0 else last_draws/last_played
