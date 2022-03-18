class StatisticsEngine:
    def __init__(self):
        self.home_column = None
        self.away_column = None
        self._home_goals_column = None
        self._away_goals_column = None
        self._result_column = None
        self._last_n = None
        self._goal_diffs_margin = None

    def set_options(
            self,
            home_column,
            away_column,
            home_goals_column,
            away_goals_column,
            result_column,
            last_n,
            goal_diffs_margin
    ):
        self.home_column = home_column
        self.away_column = away_column
        self._home_goals_column = home_goals_column
        self._away_goals_column = away_goals_column
        self._result_column = result_column
        self._last_n = last_n
        self._goal_diffs_margin = goal_diffs_margin

    def compute_last_wins(self, **kwargs):
        previous_matches = kwargs['previous_matches']
        team_name = kwargs['team_name']
        is_home = kwargs['is_home']

        wins = 0
        n = self._last_n
        team_column = self.home_column if is_home else self.away_column

        for match in previous_matches:
            if match[team_column] == team_name:
                match_result = match[self._result_column]

                if is_home:
                    if match_result == 'H':
                        wins += 1
                else:
                    if match_result == 'A':
                        wins += 1
                n -= 1
                if n == 0:
                    break

        if n > 0:
            return None
        return wins

    def compute_last_losses(self, **kwargs):
        previous_matches = kwargs['previous_matches']
        team_name = kwargs['team_name']
        is_home = kwargs['is_home']

        losses = 0
        n = self._last_n
        team_column = self.home_column if is_home else self.away_column

        for match in previous_matches:
            if match[team_column] == team_name:
                match_result = match[self._result_column]

                if is_home:
                    if match_result == 'A':
                        losses += 1
                else:
                    if match_result == 'H':
                        losses += 1
                n -= 1
                if n == 0:
                    break
        if n > 0:
            return None
        return losses

    def compute_last_goal_forward(self, **kwargs):
        previous_matches = kwargs['previous_matches']
        team_name = kwargs['team_name']
        is_home = kwargs['is_home']

        goal_forward = 0
        n = self._last_n
        team_column = self.home_column if is_home else self.away_column

        for match in previous_matches:
            if match[team_column] == team_name:
                if is_home:
                    goal_forward += match[self._home_goals_column]
                else:
                    goal_forward += match[self._away_goals_column]

                n -= 1
                if n == 0:
                    break

        if n > 0:
            return None
        return goal_forward

    def compute_last_goal_diff_wins(self, **kwargs):
        previous_matches = kwargs['previous_matches']
        team_name = kwargs['team_name']
        is_home = kwargs['is_home']

        goal_diff_wins = 0
        n = self._last_n
        team_column = self.home_column if is_home else self.away_column

        for match in previous_matches:
            if match[team_column] == team_name:
                home_goals = match[self._home_goals_column]
                away_goals = match[self._away_goals_column]

                if is_home:
                    if home_goals >= away_goals + self._goal_diffs_margin:
                        goal_diff_wins += 1
                else:
                    if away_goals >= home_goals + self._goal_diffs_margin:
                        goal_diff_wins += 1

                n -= 1
                if n == 0:
                    break

        if n > 0:
            return None
        return goal_diff_wins

    def compute_last_goal_diff_losses(self, **kwargs):
        previous_matches = kwargs['previous_matches']
        team_name = kwargs['team_name']
        is_home = kwargs['is_home']

        goal_diff_losses = 0
        n = self._last_n
        team_column = self.home_column if is_home else self.away_column

        for match in previous_matches:
            if match[team_column] == team_name:
                home_goals = match[self._home_goals_column]
                away_goals = match[self._away_goals_column]

                if is_home:
                    if home_goals + self._goal_diffs_margin <= away_goals:
                        goal_diff_losses += 1
                else:
                    if away_goals + self._goal_diffs_margin <= home_goals:
                        goal_diff_losses += 1

                n -= 1
                if n == 0:
                    break

        if n > 0:
            return None
        return goal_diff_losses

    def compute_last_goal_diffs(self, **kwargs):
        previous_matches = kwargs['previous_matches']
        team_name = kwargs['team_name']
        is_home = kwargs['is_home']

        goals_scored = 0
        goals_received = 0
        n = self._last_n
        team_column = self.home_column if is_home else self.away_column

        for match in previous_matches:
            if match[team_column] == team_name:
                home_goals = match[self._home_goals_column]
                away_goals = match[self._away_goals_column]

                if is_home:
                    goals_scored += home_goals
                    goals_received += away_goals
                else:
                    goals_received += home_goals
                    goals_scored += away_goals

                n -= 1
                if n == 0:
                    break

        if n > 0:
            return None
        return goals_scored - goals_received

    def compute_total_win_rate(self, **kwargs):
        previous_matches = kwargs['previous_matches']
        team_name = kwargs['team_name']
        is_home = kwargs['is_home']

        wins = 0
        total_matches = 0
        team_column = self.home_column if is_home else self.away_column

        for match in previous_matches:
            if match[team_column] == team_name:
                match_result = match[self._result_column]

                if is_home:
                    if match_result == 'H':
                        wins += 1
                else:
                    if match_result == 'A':
                        wins += 1
                total_matches += 1

        if total_matches == 0:
            return None
        return round(wins / total_matches, 2)

    def compute_total_win_draw_rate(self, **kwargs):
        previous_matches = kwargs['previous_matches']
        team_name = kwargs['team_name']
        is_home = kwargs['is_home']

        wins_draws = 0
        total_matches = 0
        team_column = self.home_column if is_home else self.away_column

        for match in previous_matches:
            if match[team_column] == team_name:
                match_result = match[self._result_column]

                if match_result == 'D':
                    wins_draws += 1
                else:
                    if is_home:
                        if match_result == 'H':
                            wins_draws += 1
                    else:
                        if match_result == 'A':
                            wins_draws += 1
                total_matches += 1

        if total_matches == 0:
            return None
        return round(wins_draws / total_matches, 2)
