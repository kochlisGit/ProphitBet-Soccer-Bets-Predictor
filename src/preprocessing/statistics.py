import pandas as pd
from functools import reduce
from typing import List
from tqdm import tqdm


class StatisticsEngine:
    """ The statistics engine that calculates match statistics, features and performance. """

    def __init__(self, match_history_window: int, goal_diff_margin: int):
        """
            :param match_history_window: The window length of match history to calculate the statistics.
            :param goal_diff_margin: The minimum goal difference in a match to consider it as an outstanding
                                     performance for the winners.
        """

        self._n = match_history_window
        self._gd_margin = goal_diff_margin

        self._basic_stats_fn = {
            'HW': self._compute_home_wins,
            'AW': self._compute_away_wins,
            'HL': self._compute_home_losses,
            'AL': self._compute_away_losses,
            'HGF': self._compute_home_goals_forward,
            'AGF': self._compute_away_goals_forward,
            'HAGF': self._compute_home_away_goals_forward,
            'HGA': self._compute_home_goals_against,
            'AGA': self._compute_away_goals_against,
            'HAGA': self._compute_home_away_goals_against,
            'HGD': self._compute_home_goal_diff,
            'AGD': self._compute_away_goal_diff,
            'HAGD': self._compute_home_away_goal_diff,
            'HWGD': self._compute_home_wins_margin_goal_diff,
            'AWGD': self._compute_away_wins_margin_goal_diff,
            'HAWGD': self._compute_home_away_wins_margin_goal_diff,
            'HLGD': self._compute_home_losses_margin_goal_diff,
            'ALGD': self._compute_away_losses_margin_goal_diff,
            'HALGD': self._compute_home_away_losses_margin_goal_diff,
            'HW%': self._compute_total_home_win_rate,
            'HL%': self._compute_total_home_loss_rate,
            'AW%': self._compute_total_away_win_rate,
            'AL%': self._compute_total_away_loss_rate
        }
        self._extended_stats_fn = {
            'HSTF': self._compute_home_shots_on_target_forward,
            'ASTF': self._compute_away_shots_on_target_forward,
            'HCF': self._compute_home_corners_forward,
            'ACF': self._compute_away_corners_forward
        }
        self._all_stats_fn = {
            **self._basic_stats_fn,
            **self._extended_stats_fn
        }

        self._basic_stat_columns = list(self._basic_stats_fn.keys())
        self._extra_stat_columns = list(self._extended_stats_fn.keys())
        self._all_stat_columns = list(self._all_stats_fn.keys())

    @staticmethod
    def get_basic_stat_columns() -> List[str]:
        return [
            'HW', 'AW', 'HL', 'AL', 'HGF', 'AGF', 'HAGF', 'HGA', 'AGA', 'HAGA', 'HGD', 'AGD', 'HAGD',
            'HWGD', 'AWGD', 'HAWGD', 'HLGD', 'ALGD', 'HALGD', 'HW%', 'HL%', 'AW%', 'AL%'
        ]

    @staticmethod
    def get_extended_stat_columns() -> List[str]:
        return ['HSTF', 'ASTF', 'HCF', 'ACF']

    def compute_stats(self, df: pd.DataFrame, stat_columns: List[str]) -> pd.DataFrame:
        """ Computes the requested statistic columns for each season. Finally, it sorts matches in descending order. """

        # Validate that the matches are provided in ascending order, otherwise the calculations will be wrong.
        if not df['Date'].is_monotonic_increasing:
            raise ValueError('Expected dates to be sorted in a ascending order.')

        stat_funcs = [self._all_stats_fn[col] for col in stat_columns]

        def season_pipeline(season_df: pd.DataFrame) -> pd.DataFrame:
            """ Calculates all requested statistics for each season. """

            return reduce(lambda s_df, fn: fn(s_df), stat_funcs, season_df)

        tqdm.pandas(desc='Processing Season')
        df = df.groupby(by='Season', group_keys=False).progress_apply(season_pipeline)

        # Sort matches by descending order and return dataframe.
        return df.sort_values(by=['Date', 'Home'], ascending=False)

    def _aggregate_previous_stats(self, match_stats: pd.Series) -> pd.Series:
        """ Shifts 1 index to avoid current match, rolls last N matches and sums the provided stats. """

        return match_stats.shift(periods=1).rolling(window=self._n, min_periods=self._n).sum()

    def _compute_home_wins(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes last N home wins. """

        temp_df = df[['Home', 'Result']]
        temp_df['HomeWin'] = temp_df['Result'].eq('H').astype(int)
        df['HW'] = temp_df.groupby(by='Home')['HomeWin'].transform(self._aggregate_previous_stats)
        return df

    def _compute_away_wins(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes last N away wins. """

        temp_df = df[['Away', 'Result']]
        temp_df['AwayWin'] = temp_df['Result'].eq('A').astype(int)
        df['AW'] = temp_df.groupby(by='Away')['AwayWin'].transform(self._aggregate_previous_stats)
        return df

    def _compute_home_losses(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes last N home losses. """

        temp_df = df[['Home', 'Result']]
        temp_df['HomeLoss'] = temp_df['Result'].eq('A').astype(int)
        df['HL'] = temp_df.groupby(by='Home')['HomeLoss'].transform(self._aggregate_previous_stats)
        return df

    def _compute_away_losses(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes last N away losses. """

        temp_df = df[['Away', 'Result']]
        temp_df['AwayLoss'] = temp_df['Result'].eq('H').astype(int)
        df['AL'] = temp_df.groupby(by='Away')['AwayLoss'].transform(self._aggregate_previous_stats)
        return df

    def _compute_home_goals_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Aggregates last N goals scored as home team. """

        df['HGF'] = df.groupby(by='Home')['HG'].transform(self._aggregate_previous_stats)
        return df

    def _compute_away_goals_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Aggregates last N goals scored as away team. """

        df['AGF'] = df.groupby(by='Away')['AG'].transform(self._aggregate_previous_stats)
        return df

    def _compute_home_away_goals_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes difference between HGF and AGF. """

        if 'HGF' in df and 'AGF' in df:
            df['HAGF'] = df['HGF'] - df['AGF']
            return df

        temp_df = df[['Home', 'Away', 'HG', 'AG']]
        temp_df['HGF'] = df.groupby(by='Home')['HG'].transform(self._aggregate_previous_stats)
        temp_df['AGF'] = df.groupby(by='Away')['AG'].transform(self._aggregate_previous_stats)
        df['HAGF'] = temp_df['HGF'] - temp_df['AGF']
        return df

    def _compute_home_goals_against(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Aggregates last N goals that home team received. """

        df['HGA'] = df.groupby(by='Home')['AG'].transform(self._aggregate_previous_stats)
        return df

    def _compute_away_goals_against(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Aggregates last N goals that away team received. """

        df['AGA'] = df.groupby(by='Away')['HG'].transform(self._aggregate_previous_stats)
        return df

    def _compute_home_away_goals_against(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes difference between HGA and AGA. """

        if 'HGA' in df and 'AGA' in df:
            df['HAGA'] = df['HGA'] - df['AGA']
            return df

        temp_df = df[['Home', 'Away', 'HG', 'AG']]
        temp_df['HGA'] = df.groupby(by='Home')['AG'].transform(self._aggregate_previous_stats)
        temp_df['AGA'] = df.groupby(by='Away')['HG'].transform(self._aggregate_previous_stats)
        df['HAGA'] = temp_df['HGA'] - temp_df['AGA']
        return df

    def _compute_home_goal_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes last home goal differences as HGD = HGF - HGA """

        # If both HGF and HGA exist, uses pre-computed statistics.
        if 'HGF' in df and 'HGA' in df:
            df['HGD'] = df['HGF'] - df['HGA']
            return df

        # Calculate HG - AG and aggregate the results.
        temp_df = df[['Home', 'HG', 'AG']]
        temp_df['HG-AG'] = temp_df['HG'] - temp_df['AG']
        df['HGD'] = temp_df.groupby(by='Home')['HG-AG'].transform(self._aggregate_previous_stats)
        return df

    def _compute_away_goal_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes last away goal differences as AGD = AGF - AGA """

        # If both AGF and AGA exist, uses pre-computed statistics.
        if 'AGF' in df and 'AGA' in df:
            df['AGD'] = df['AGF'] - df['AGA']
            return df

        # Calculate HG - AG and aggregate the results.
        temp_df = df[['Away', 'HG', 'AG']]
        temp_df['AG-HG'] = temp_df['AG'] - temp_df['HG']
        df['AGD'] = temp_df.groupby(by='Away')['AG-HG'].transform(self._aggregate_previous_stats)
        return df

    def _compute_home_away_goal_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes difference between HGD and AGD """

        if 'HGD' in df and 'AGD' in df:
            df['HAGD'] = df['HGD'] - df['AGD']
            return df

        temp_df = df[['Home', 'Away', 'HG', 'AG']]
        temp_df['HG-AG'] = temp_df['HG'] - temp_df['AG']
        temp_df['AG-HG'] = temp_df['AG'] - temp_df['HG']
        temp_df['HGD'] = temp_df.groupby(by='Home')['HG-AG'].transform(self._aggregate_previous_stats)
        temp_df['AGD'] = temp_df.groupby(by='Away')['AG-HG'].transform(self._aggregate_previous_stats)
        df['HAGD'] = temp_df['HGD'] - temp_df['AGD']
        return df

    def _compute_home_wins_margin_goal_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes last N matches where home team won with outstanding performance. """

        temp_df = df[['Home', 'HG', 'AG']]
        temp_df['HomeWinMargin'] = (temp_df['HG'] - temp_df['AG']).ge(self._gd_margin).astype(int)
        df['HWGD'] = temp_df.groupby(by='Home')['HomeWinMargin'].transform(self._aggregate_previous_stats)
        return df

    def _compute_away_wins_margin_goal_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes last N matches where away team won with outstanding performance. """

        temp_df = df[['Away', 'HG', 'AG']]
        temp_df['AwayWinMargin'] = (temp_df['AG'] - temp_df['HG']).ge(self._gd_margin).astype(int)
        df['AWGD'] = temp_df.groupby(by='Away')['AwayWinMargin'].transform(self._aggregate_previous_stats)
        return df

    def _compute_home_away_wins_margin_goal_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes difference between HWGD and AWGD """

        if 'HWGD' in df and 'AWGD' in df:
            df['HAWGD'] = df['HWGD'] - df['AWGD']
            return df

        temp_df = df[['Home', 'Away', 'HG', 'AG']]
        temp_df['HomeWinMargin'] = (temp_df['HG'] - temp_df['AG']).ge(self._gd_margin).astype(int)
        temp_df['AwayWinMargin'] = (temp_df['AG'] - temp_df['HG']).ge(self._gd_margin).astype(int)
        temp_df['HWGD'] = temp_df.groupby(by='Home')['HomeWinMargin'].transform(self._aggregate_previous_stats)
        temp_df['AWGD'] = temp_df.groupby(by='Away')['AwayWinMargin'].transform(self._aggregate_previous_stats)
        df['HAWGD'] = temp_df['HWGD'] - df['AWGD']
        return df

    def _compute_home_losses_margin_goal_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes last N matches where home team lost with away team having outstanding performance. """

        temp_df = df[['Home', 'HG', 'AG']]
        temp_df['HomeLossMargin'] = (temp_df['AG'] - temp_df['HG']).ge(self._gd_margin).astype(int)
        df['HLGD'] = temp_df.groupby(by='Home')['HomeLossMargin'].transform(self._aggregate_previous_stats)
        return df

    def _compute_away_losses_margin_goal_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes last N matches where away team lost with home team having outstanding performance. """

        temp_df = df[['Away', 'HG', 'AG']]
        temp_df['AwayLossMargin'] = (temp_df['HG'] - temp_df['AG']).ge(self._gd_margin).astype(int)
        df['ALGD'] = temp_df.groupby(by='Away')['AwayLossMargin'].transform(self._aggregate_previous_stats)
        return df

    def _compute_home_away_losses_margin_goal_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes difference between HLGD and ALGD """

        if 'HLGD' in df and 'ALGD' in df:
            df['HALGD'] = df['HLGD'] - df['ALGD']
            return df

        temp_df = df[['Home', 'Away', 'HG', 'AG']]
        temp_df['HomeLossMargin'] = (temp_df['AG'] - temp_df['HG']).ge(self._gd_margin).astype(int)
        temp_df['AwayLossMargin'] = (temp_df['HG'] - temp_df['AG']).ge(self._gd_margin).astype(int)
        temp_df['HLGD'] = temp_df.groupby(by='Home')['HomeLossMargin'].transform(self._aggregate_previous_stats)
        temp_df['ALGD'] = temp_df.groupby(by='Away')['AwayLossMargin'].transform(self._aggregate_previous_stats)
        df['HALGD'] = temp_df['HLGD'] - df['ALGD']
        return df

    def _compute_total_home_win_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes total win rate for home team from the beginning of the league. """

        temp_df = df[['Home', 'Result']]
        temp_df['HomeWins'] = temp_df['Result'].eq('H').astype(float)
        temp_df['CumWins'] = temp_df.groupby(by='Home')['HomeWins'].cumsum() - temp_df['HomeWins']
        temp_df['CumCounts'] = temp_df.groupby(by='Home').cumcount()
        df['HW%'] = (temp_df['CumWins']/temp_df['CumCounts']*100).round(decimals=1)
        return df

    def _compute_total_home_loss_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes total loss rate for home team from the beginning of the league. """

        temp_df = df[['Home', 'Result']]
        temp_df['HomeLosses'] = temp_df['Result'].eq('A').astype(float)
        temp_df['CumLosses'] = temp_df.groupby(by='Home')['HomeLosses'].cumsum() - temp_df['HomeLosses']
        temp_df['CumCounts'] = temp_df.groupby(by='Home').cumcount()
        df['HL%'] = (temp_df['CumLosses']/temp_df['CumCounts']*100).round(decimals=1)
        return df

    def _compute_total_away_win_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes total win rate for away team from the beginning of the league. """

        temp_df = df[['Away', 'Result']]
        temp_df['AwayWins'] = temp_df['Result'].eq('A').astype(float)
        temp_df['CumWins'] = temp_df.groupby(by='Away')['AwayWins'].cumsum() - temp_df['AwayWins']
        temp_df['CumCounts'] = temp_df.groupby(by='Away').cumcount()
        df['AW%'] = (temp_df['CumWins']/temp_df['CumCounts']*100).round(decimals=1)
        return df

    def _compute_total_away_loss_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes total loss rate for away team from the beginning of the league. """

        temp_df = df[['Away', 'Result']]
        temp_df['AwayLosses'] = temp_df['Result'].eq('H').astype(float)
        temp_df['CumLosses'] = temp_df.groupby(by='Away')['AwayLosses'].cumsum() - temp_df['AwayLosses']
        temp_df['CumCounts'] = temp_df.groupby(by='Away').cumcount()
        df['AL%'] = (temp_df['CumLosses']/temp_df['CumCounts']*100).round(decimals=1)
        return df

    def _compute_home_shots_on_target_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Aggregates last N shots on targets for home team. """

        df['HSTF'] = df.groupby(by='Home')['HST'].transform(self._aggregate_previous_stats)
        return df

    def _compute_away_shots_on_target_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Aggregates last N shots on targets for away team. """

        df['ASTF'] = df.groupby(by='Away')['AST'].transform(self._aggregate_previous_stats)
        return df

    def _compute_home_corners_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Aggregates last N corners for home team. """

        df['HCF'] = df.groupby(by='Home')['HC'].transform(self._aggregate_previous_stats)
        return df

    def _compute_away_corners_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Aggregates last N corners for away team. """

        df['ACF'] = df.groupby(by='Away')['AC'].transform(self._aggregate_previous_stats)
        return df
