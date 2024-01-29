import pandas as pd


class StatisticsEngine:
    def __init__(self, match_history_window: int, goal_diff_margin: int):
        self._match_history_window = match_history_window
        self._goal_diff_margin = goal_diff_margin

        self._statistic_features = {
            'HW': self.compute_home_wins,
            'HL': self.compute_home_losses,
            'AW': self.compute_away_wins,
            'AL': self.compute_away_losses,
            'HGF': self.compute_home_goals_forward,
            'HGA': self.compute_home_goals_against,
            'AGF': self.compute_away_goals_forward,
            'AGA': self.compute_away_goals_against,
            'HWGD': self.compute_home_wins_margin_goal_diff,
            'HLGD': self.compute_home_losses_margin_goal_diff,
            'AWGD': self.compute_away_wins_margin_goal_diff,
            'ALGD': self.compute_away_losses_margin_goal_diff,
            'HW%': self.compute_total_home_win_rate,
            'HL%': self.compute_total_home_loss_rate,
            'AW%': self.compute_total_away_win_rate,
            'AL%': self.compute_total_away_loss_rate
        }

    def _compute_last_results(self, matches_df: pd.DataFrame, team_column: str, result: str) -> pd.Series:
        def compute_results(results: pd.Series) -> pd.Series:
            window = pd.api.indexers.FixedForwardWindowIndexer(window_size=self._match_history_window)
            return results.eq(result).rolling(window=window, min_periods=self._match_history_window).sum().shift(-1)

        return matches_df.groupby(['Season', team_column], sort=False)['Result'].apply(compute_results)

    def compute_home_wins(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        matches_df['HW'] = self._compute_last_results(matches_df=matches_df, team_column='Home Team', result='H')
        return matches_df

    def compute_home_losses(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        matches_df['HL'] = self._compute_last_results(matches_df=matches_df, team_column='Home Team', result='A')
        return matches_df

    def compute_away_wins(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        matches_df['AW'] = self._compute_last_results(matches_df=matches_df, team_column='Away Team', result='A')
        return matches_df

    def compute_away_losses(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        matches_df['AL'] = self._compute_last_results(matches_df=matches_df, team_column='Away Team', result='H')
        return matches_df

    def _compute_last_goals(self, matches_df: pd.DataFrame, team_column: str, forward: bool) -> pd.Series:
        def compute_results(results: pd.Series) -> pd.Series:
            window = pd.api.indexers.FixedForwardWindowIndexer(window_size=self._match_history_window)
            return results.rolling(window=window, min_periods=self._match_history_window).sum().shift(-1)

        col = 'HG' if forward else 'AG'
        return matches_df.groupby(['Season', team_column], sort=False)[col].apply(compute_results)

    def compute_home_goals_forward(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        matches_df['HGF'] = self._compute_last_goals(matches_df=matches_df, team_column='Home Team', forward=True)
        return matches_df

    def compute_home_goals_against(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        matches_df['HGA'] = self._compute_last_goals(matches_df=matches_df, team_column='Home Team', forward=False)
        return matches_df

    def compute_away_goals_forward(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        matches_df['AGF'] = self._compute_last_goals(matches_df=matches_df, team_column='Away Team', forward=False)
        return matches_df

    def compute_away_goals_against(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        matches_df['AGA'] = self._compute_last_goals(matches_df=matches_df, team_column='Away Team', forward=True)
        return matches_df

    def _compute_last_results_with_margin(self, matches_df: pd.DataFrame, team_column: str, result: str) -> pd.Series:
        def compute_results(df: pd.DataFrame) -> pd.Series:
            goal_diffs = df['HG'] - df['AG']
            margin_results = goal_diffs - self._goal_diff_margin >= 0 if result == 'H' else goal_diffs + self._goal_diff_margin <= 0
            window = pd.api.indexers.FixedForwardWindowIndexer(window_size=self._match_history_window)
            return margin_results.rolling(
                window=window,
                min_periods=self._match_history_window
            ).sum().shift(-1).to_frame()

        return matches_df.groupby(['Season', team_column], sort=False)[['HG', 'AG']].apply(compute_results)

    def compute_home_wins_margin_goal_diff(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        matches_df['HWGD'] = self._compute_last_results_with_margin(matches_df=matches_df, team_column='Home Team', result='H')
        return matches_df

    def compute_home_losses_margin_goal_diff(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        matches_df['HLGD'] = self._compute_last_results_with_margin(matches_df=matches_df, team_column='Home Team', result='A')
        return matches_df

    def compute_away_wins_margin_goal_diff(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        matches_df['AWGD'] = self._compute_last_results_with_margin(matches_df=matches_df, team_column='Away Team', result='A')
        return matches_df

    def compute_away_losses_margin_goal_diff(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        matches_df['ALGD'] = self._compute_last_results_with_margin(matches_df=matches_df, team_column='Away Team', result='H')
        return matches_df

    def _compute_total_result_rate(self, matches_df: pd.DataFrame, team_column: str, result: str) -> pd.Series:
        def compute_results(results: pd.Series) -> pd.Series:
            window = pd.api.indexers.FixedForwardWindowIndexer(window_size=results.shape[0])
            return results.eq(result).rolling(
                window=window,
                min_periods=self._match_history_window
            ).mean().round(decimals=2).shift(-1)

        return matches_df.groupby(['Season', team_column], sort=False)['Result'].apply(compute_results)

    def compute_total_home_win_rate(self, matches_df: pd.DataFrame):
        matches_df['HW%'] = self._compute_total_result_rate(matches_df=matches_df, team_column='Home Team', result='H')
        return matches_df

    def compute_total_home_loss_rate(self, matches_df: pd.DataFrame):
        matches_df['HL%'] = self._compute_total_result_rate(matches_df=matches_df, team_column='Home Team', result='A')
        return matches_df

    def compute_total_away_win_rate(self, matches_df: pd.DataFrame):
        matches_df['AW%'] = self._compute_total_result_rate(matches_df=matches_df, team_column='Away Team', result='A')
        return matches_df

    def compute_total_away_loss_rate(self, matches_df: pd.DataFrame):
        matches_df['AL%'] = self._compute_total_result_rate(matches_df=matches_df, team_column='Away Team', result='H')
        return matches_df

    def compute_statistics(self, matches_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        for col in features:
            if col in self._statistic_features:
                self._statistic_features[col](matches_df=matches_df)
        return matches_df
