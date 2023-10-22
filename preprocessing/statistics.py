import numpy as np
import pandas as pd


class StatisticsEngine:
    Columns = [
        'HW', 'HL', 'HD', 'HGF', 'HGA', 'HGDW', 'HGDL', 'HW%', 'HD%',
        'AW', 'AL', 'AD', 'AGF', 'AGA', 'AGDW', 'AGDL', 'AW%', 'AD%',
    ]

    def __init__(
            self,
            matches_df: pd.DataFrame,
            last_n_matches: int,
            goal_diff_margin: int
    ):
        self._matches_df = matches_df
        self._match_history = matches_df[['Season', 'Home Team', 'Away Team', 'HG', 'AG', 'Result']].values
        self._max_season = self._match_history[0, 0]
        self._min_season = self._match_history[-1, 0]

        self._last_n_matches = last_n_matches
        self._goal_diff_margin = goal_diff_margin

        self._statistics_mapper = {
            'HW': self._compute_last_n_home_wins,
            'HL': self._compute_last_n_home_losses,
            'HD': self._compute_last_n_home_draws,
            'HGF': self._compute_last_n_home_goals_forward,
            'HGA': self._compute_last_n_home_goals_against,
            'HGDW': self._compute_last_n_home_wins_goals_diff,
            'HGDL': self._compute_last_n_home_losses_goals_diff,
            'HW%': self._compute_total_home_win_rate,
            'HD%': self._compute_total_home_draw_rate,
            'AW': self._compute_last_n_away_wins,
            'AL': self._compute_last_n_away_losses,
            'AD': self._compute_last_n_away_draws,
            'AGF': self._compute_last_n_away_goals_forward,
            'AGA': self._compute_last_n_away_goals_against,
            'AGDW': self._compute_last_n_away_wins_goals_diff,
            'AGDL': self._compute_last_n_away_losses_goals_diff,
            'AW%': self._compute_total_home_win_rate,
            'AD%': self._compute_total_home_draw_rate
        }

    def compute_statistics(
            self,
            statistic_columns: list
    ) -> pd.DataFrame:
        matches_df = self._matches_df.copy()

        for column in statistic_columns:
            matches_df[column] = self._statistics_mapper[column]()
        matches_df = matches_df.dropna()
        matches_df[statistic_columns] = matches_df[statistic_columns].astype(dtype=np.int32)
        return matches_df

    def _compute_last_n_home_wins(self) -> pd.Series:
        return self._compute_last_results(team_index=1, target_result_value='H')

    def _compute_last_n_home_losses(self) -> pd.Series:
        return self._compute_last_results(team_index=1, target_result_value='A')

    def _compute_last_n_away_wins(self) -> pd.Series:
        return self._compute_last_results(team_index=2, target_result_value='A')

    def _compute_last_n_away_losses(self) -> pd.Series:
        return self._compute_last_results(team_index=2, target_result_value='H')
    
    def _compute_last_n_home_draws(self) -> pd.Series:
        return self._compute_last_results(team_index=1, target_result_value='D')

    def _compute_last_n_away_draws(self) -> pd.Series:
        return self._compute_last_results(team_index=2, target_result_value='D')

    def _compute_last_results(self, team_index: int, target_result_value: str):
        last_results = []

        for season in range(self._max_season, self._min_season - 1, -1):
            match_history = self._match_history[self._match_history[:, 0] == season]

            for i, match in enumerate(match_history):
                team_name = match[team_index]
                target_results = 0
                last_n = 0

                for previous_match in match_history[i + 1:]:
                    if previous_match[team_index] == team_name:
                        result = previous_match[5]
                        last_n += 1

                        if result == target_result_value:
                            target_results += 1
                    if last_n == self._last_n_matches:
                        break
                last_results.append(target_results if last_n == self._last_n_matches else np.nan)
        return pd.Series(last_results)

    def _compute_last_n_home_goals_forward(self) -> pd.Series:
        return self._compute_last_goals(team_index=1, goals_index=3)

    def _compute_last_n_home_goals_against(self) -> pd.Series:
        return self._compute_last_goals(team_index=1, goals_index=4)

    def _compute_last_n_away_goals_forward(self) -> pd.Series:
        return self._compute_last_goals(team_index=2, goals_index=4)

    def _compute_last_n_away_goals_against(self) -> pd.Series:
        return self._compute_last_goals(team_index=2, goals_index=3)

    def _compute_last_goals(self, team_index: int, goals_index: int):
        last_goals = []

        for season in range(self._max_season, self._min_season - 1, -1):
            match_history = self._match_history[self._match_history[:, 0] == season]

            for i, match in enumerate(match_history):
                team_name = match[team_index]
                goals = 0
                last_n = 0

                for previous_match in match_history[i + 1:]:
                    if previous_match[team_index] == team_name:
                        goals += previous_match[goals_index]
                        last_n += 1
                    if last_n == self._last_n_matches:
                        break
                last_goals.append(goals if last_n == self._last_n_matches else np.nan)
        return pd.Series(last_goals)

    def _compute_last_n_home_wins_goals_diff(self) -> pd.Series:
        return self._compute_last_results_with_goals_diff(
            team_index=1, target_higher_goals_index=3, target_lower_goals_index=4
        )

    def _compute_last_n_home_losses_goals_diff(self) -> pd.Series:
        return self._compute_last_results_with_goals_diff(
            team_index=1, target_higher_goals_index=4, target_lower_goals_index=3
        )

    def _compute_last_n_away_wins_goals_diff(self) -> pd.Series:
        return self._compute_last_results_with_goals_diff(
            team_index=2, target_higher_goals_index=4, target_lower_goals_index=3
        )

    def _compute_last_n_away_losses_goals_diff(self) -> pd.Series:
        return self._compute_last_results_with_goals_diff(
            team_index=2, target_higher_goals_index=3, target_lower_goals_index=4
        )

    def _compute_last_results_with_goals_diff(
            self,
            team_index: int,
            target_higher_goals_index: int,
            target_lower_goals_index: int
    ):
        last_results_with_goals_diff = []

        for season in range(self._max_season, self._min_season - 1, -1):
            match_history = self._match_history[self._match_history[:, 0] == season]

            for i, match in enumerate(match_history):
                team_name = match[team_index]
                target_results_with_goals_diff = 0
                last_n = 0

                for previous_match in match_history[i + 1:]:
                    if previous_match[team_index] == team_name:
                        target_higher_goals = previous_match[target_higher_goals_index]
                        target_lower_goals = previous_match[target_lower_goals_index]
                        last_n += 1

                        if target_higher_goals - target_lower_goals >= self._goal_diff_margin:
                            target_results_with_goals_diff += 1
                    if last_n == self._last_n_matches:
                        break
                last_results_with_goals_diff.append(
                    target_results_with_goals_diff if last_n == self._last_n_matches else np.nan
                )
        return pd.Series(last_results_with_goals_diff)

    def _compute_total_home_win_rate(self) -> pd.Series:
        return self._compute_total_results_rate(team_index=1, target_result_value='H')

    def _compute_total_home_draw_rate(self) -> pd.Series:
        return self._compute_total_results_rate(team_index=1, target_result_value='D')

    def _compute_total_away_win_rate(self) -> pd.Series:
        return self._compute_total_results_rate(team_index=2, target_result_value='A')

    def _compute_total_away_loss_rate(self) -> pd.Series:
        return self._compute_total_results_rate(team_index=2, target_result_value='D')

    def _compute_total_results_rate(self, team_index: int, target_result_value: str):
        last_result_rates = []

        for season in range(self._max_season, self._min_season - 1, -1):
            match_history = self._match_history[self._match_history[:, 0] == season]

            for i, match in enumerate(match_history):
                team_name = match[team_index]
                target_results = 0
                non_target_results = 0

                for previous_match in match_history[i + 1:]:
                    if previous_match[team_index] == team_name:
                        result = previous_match[5]

                        if result == target_result_value:
                            target_results += 1
                        else:
                            non_target_results += 1
                total_results = target_results + non_target_results
                last_result_rates.append(np.nan if total_results == 0 else round(target_results*100/total_results))
        return pd.Series(last_result_rates)
