import pandas as pd
from typing import List, Set
from tqdm import tqdm


class StatisticsEngine:
    """ The statistics engine that calculates match statistics, features and performance.

        Attributes
        ----------
        :param match_history_window: (int) window length of match history to calculate the statistics. Defaults to 3.
        :param goal_diff_margin: (int) minimum goal difference for marginal win/loss. Defaults to 2.
        :param aggregation: (str) aggregation method to calculate the statistics. Supports "sum" and "mean".
                            Defaults to 'sum'.
    """

    def __init__(self, match_history_window: int = 3, goal_diff_margin: int = 2, aggregation: str = 'sum'):
        self._margin = goal_diff_margin

        # Helper method that computes rolling sum of the provided stats.
        if aggregation == 'sum':
            self._aggregate = lambda values: \
                values.shift(periods=1).rolling(window=match_history_window, min_periods=match_history_window).sum()
        elif aggregation == 'mean':
            self._aggregate = lambda values: \
                values.shift(periods=1).rolling(window=match_history_window, min_periods=match_history_window).mean()
        else:
            raise NotImplementedError(f'Not implemented aggregation method: "{aggregation}".')

        self._home_stats = {
            'HW': self._hw,             # Recent home wins as home.
            'HL': self._hl,             # Recent home losses as home.
            'HGF': self._hgf,           # Recent home goals forward as home.
            'HGA': self._hga,           # Recent home goals against as home.
            'HGD': self._hgd,           # HGF - HGA.
            'HWGD': self._hwgd,         # Recent home marginal wins as home.
            'HLGD': self._hlgd,         # Recent home marginal losses as home.
            'HW%': self._hwp,           # Total home win percentage as home.
            'HL%': self._hlp            # Total home loss percentage as home.
        }
        self._away_stats = {
            'AW': self._aw,             # Recent away wins as away.
            'AL': self._al,             # Recent away losses as away.
            'AGF': self._agf,           # Recent away goals forward as away.
            'AGA': self._aga,           # Recent away goals against as away.
            'AGD': self._agd,           # AGF - AGA.
            'AWGD': self._awgd,         # Recent away marginal wins as away.
            'ALGD': self._algd,         # Recent away marginal losses as away.
            'AW%': self._awp,           # Total away win percentage as away.
            'AL%': self._alp            # Total away loss percentage as away.
        }
        self._home_away_stats = {
            'HAGF': self._hagf,         # HGF - AGF.
            'HAGA': self._haga,         # HGA - AGA.
            'HAGD': self._hagd,         # HGD - AGD.
            'HAWGD': self._hawgd,       # HWGD - AWGD.
            'HALGD': self._halgd,       # HLGD - ALGD.
            'HAGFT': self._hagft,       # HGFT - AGFT.
            'HAGAT': self._hagat        # HGAT - AGAT.
        }
        self._total_stats = {
            'HWT': self._hwt,           # Recent home wins both as home/away.
            'HLT': self._hlt,           # Recent home losses both as home/away.
            'HGFT': self._hgft,         # Recent home goals forward both as home/away.
            'HGAT': self._hgat,         # Recent home goals against both as home/away.
            'HWT%': self._hwtp,         # Total home win percentage both as home/away.
            'HLT%': self._hltp,         # Total home loss percentage both as home/away.
            'AWT': self._awt,           # Recent away wins both as home/away.
            'ALT': self._alt,           # Recent away losses both as home/away.
            'AGFT': self._agft,         # Recent away goals forward both as home/away.
            'AGAT': self._agat,         # Recent away goals against both as home/away.
            'AWT%': self._awtp,         # Total away win percentage both as home/away.
            'ALT%': self._altp          # Total away loss percentage both as home/away.
        }
        self._stats_columns = {
            **self._home_stats,
            **self._away_stats,
            **self._home_away_stats,
            **self._total_stats
        }
        self._cache = {}

    @property
    def stats_columns(self) -> List[str]:
        return list(self._stats_columns.keys())

    def _hw(self, df: pd.DataFrame) -> pd.Series:
        home_wins = self._cache.get('home-wins', None)
        if home_wins is None:
            home_wins = self._cache['home-wins'] = df['Result'].eq('H').astype(int)

        values = home_wins.groupby(by=[df['Season'], df['Home']], sort=False, group_keys=False).transform(self._aggregate)
        return values

    def _hl(self, df: pd.DataFrame) -> pd.Series:
        away_wins = self._cache.get('away-wins', None)
        if away_wins is None:
            away_wins = self._cache['away-wins'] = df['Result'].eq('A').astype(int)

        values = away_wins.groupby(by=[df['Season'], df['Home']], sort=False, group_keys=False).transform(self._aggregate)
        return values

    def _hgf(self, df: pd.DataFrame) -> pd.Series:
        values = self._cache['hgf'] = df['HG'].groupby(by=[df['Season'], df['Home']], sort=False, group_keys=False).transform(self._aggregate)
        return values

    def _hga(self, df: pd.DataFrame) -> pd.Series:
        values = self._cache['hga'] = df['AG'].groupby(by=[df['Season'], df['Home']], sort=False, group_keys=False).transform(self._aggregate)
        return values

    def _hgd(self, df: pd.DataFrame) -> pd.Series:
        hgf = self._cache.get('hgf', None)
        if hgf is None:
            hgf = self._cache['hgf'] = self._hgf(df=df)

        hga = self._cache.get('hga', None)
        if hga is None:
            hga = self._cache['hga'] = self._hga(df=df)

        values = self._cache['hgd'] =  hgf - hga
        return values

    def _hwgd(self, df: pd.DataFrame) -> pd.Series:
        home_margin = self._cache.get('home-margin', None)
        if home_margin is None:
            home_margin = self._cache['home-margin'] = (df['HG'] - df['AG']).ge(self._margin)

        values = self._cache['hwgd'] = home_margin.groupby(by=[df['Season'], df['Home']], sort=False, group_keys=False).transform(self._aggregate)
        return values

    def _hlgd(self, df: pd.DataFrame) -> pd.Series:
        away_margin = self._cache.get('away-margin', None)
        if away_margin is None:
            away_margin = self._cache['away-margin'] = (df['AG'] - df['HG']).ge(self._margin)

        values = self._cache['hlgd'] = away_margin.groupby(by=[df['Season'], df['Home']], sort=False, group_keys=False).transform(self._aggregate)
        return values

    def _hwp(self, df: pd.DataFrame) -> pd.Series:
        home_wins = self._cache.get('home-wins', None)
        if home_wins is None:
            home_wins = self._cache['home-wins'] = df['Result'].eq('H').astype(int)

        home_group = home_wins.groupby(by=[df['Season'], df['Home']], sort=False, group_keys=False)
        cumwins = home_group.cumsum() - home_wins
        home_cc = home_group.cumcount()
        values = (100*cumwins/home_cc).round(decimals=1)
        return values

    def _hlp(self, df: pd.DataFrame) -> pd.Series:
        away_wins = self._cache.get('away-wins', None)
        if away_wins is None:
            away_wins = self._cache['away-wins'] = df['Result'].eq('A').astype(int)

        home_group = away_wins.groupby(by=[df['Season'], df['Home']], sort=False, group_keys=False)
        cumwins = home_group.cumsum() - away_wins
        home_cc = home_group.cumcount()
        values = (100*cumwins/home_cc).round(decimals=1)
        return values

    def _aw(self, df: pd.DataFrame) -> pd.Series:
        away_wins = self._cache.get('away-wins', None)
        if away_wins is None:
            away_wins = self._cache['away-wins'] = df['Result'].eq('A').astype(int)

        values = away_wins.groupby(by=[df['Season'], df['Away']], sort=False, group_keys=False).transform(self._aggregate)
        return values

    def _al(self, df: pd.DataFrame) -> pd.Series:
        home_wins = self._cache.get('home-wins', None)
        if home_wins is None:
            home_wins = self._cache['home-wins'] = df['Result'].eq('H').astype(int)

        values = home_wins.groupby(by=[df['Season'], df['Away']], sort=False, group_keys=False).transform(self._aggregate)
        return values

    def _agf(self, df: pd.DataFrame) -> pd.Series:
        values = self._cache['agf'] = df['AG'].groupby(by=[df['Season'], df['Away']], sort=False, group_keys=False).transform(self._aggregate)
        return values

    def _aga(self, df: pd.DataFrame) -> pd.Series:
        values = self._cache['aga'] = df['HG'].groupby(by=[df['Season'], df['Away']], sort=False, group_keys=False).transform(self._aggregate)
        return values

    def _agd(self, df: pd.DataFrame) -> pd.Series:
        agf = self._cache.get('agf', None)
        if agf is None:
            agf = self._cache['agf'] = self._agf(df=df)

        aga = self._cache.get('aga', None)
        if aga is None:
            aga = self._cache['aga'] = self._aga(df=df)

        values = self._cache['agd'] =  agf - aga
        return values

    def _awgd(self, df: pd.DataFrame) -> pd.Series:
        away_margin = self._cache.get('away-margin', None)
        if away_margin is None:
            away_margin = self._cache['away-margin'] = (df['AG'] - df['HG']).ge(self._margin)

        values = self._cache['awgd'] = away_margin.groupby(by=[df['Season'], df['Away']], sort=False, group_keys=False).transform(self._aggregate)
        return values

    def _algd(self, df: pd.DataFrame) -> pd.Series:
        home_margin = self._cache.get('home-margin', None)
        if home_margin is None:
            home_margin = self._cache['home-margin'] = (df['HG'] - df['AG']).ge(self._margin)

        values = self._cache['algd'] = home_margin.groupby(by=[df['Season'], df['Away']], sort=False, group_keys=False).transform(self._aggregate)
        return values

    def _awp(self, df: pd.DataFrame) -> pd.Series:
        away_wins = self._cache.get('away-wins', None)
        if away_wins is None:
            away_wins = self._cache['away-wins'] = df['Result'].eq('A').astype(int)

        away_group = away_wins.groupby(by=[df['Season'], df['Away']], sort=False, group_keys=False)
        cumwins = away_group.cumsum() - away_wins
        away_cc = away_group.cumcount()
        values = (100*cumwins/away_cc).round(decimals=1)
        return values

    def _alp(self, df: pd.DataFrame) -> pd.Series:
        home_wins = self._cache.get('home-wins', None)
        if home_wins is None:
            home_wins = self._cache['home-wins'] = df['Result'].eq('H').astype(int)

        away_group = home_wins.groupby(by=[df['Season'], df['Away']], sort=False, group_keys=False)
        cumwins = away_group.cumsum() - home_wins
        away_cc = away_group.cumcount()
        values = (100*cumwins/away_cc).round(decimals=1)
        return values

    def _hagf(self, df: pd.DataFrame) -> pd.Series:
        hgf = self._cache.get('hgf', None)
        if hgf is None:
            hgf = self._hgf(df=df)

        agf = self._cache.get('agf', None)
        if agf is None:
            agf = self._agf(df=df)

        values = hgf - agf
        return values

    def _haga(self, df: pd.DataFrame) -> pd.Series:
        hga = self._cache.get('hga', None)
        if hga is None:
            hga = self._hga(df=df)

        aga = self._cache.get('aga', None)
        if aga is None:
            aga = self._aga(df=df)

        values = hga - aga
        return values

    def _hagd(self, df: pd.DataFrame) -> pd.Series:
        hgd = self._cache.get('hgd', None)
        if hgd is None:
            hgd = self._hgd(df=df)

        agd = self._cache.get('agd', None)
        if agd is None:
            agd = self._agd(df=df)

        values = hgd - agd
        return values

    def _hawgd(self, df: pd.DataFrame) -> pd.Series:
        hwgd = self._cache.get('hwgd', None)
        if hwgd is None:
            hwgd = self._hwgd(df=df)

        awgd = self._cache.get('awgd', None)
        if awgd is None:
            awgd = self._awgd(df=df)

        values = hwgd - awgd
        return values

    def _halgd(self, df: pd.DataFrame) -> pd.Series:
        hlgd = self._cache.get('hlgd', None)
        if hlgd is None:
            hlgd = self._hlgd(df=df)

        algd = self._cache.get('algd', None)
        if algd is None:
            algd = self._algd(df=df)

        values = hlgd - algd
        return values

    def _hagft(self, df: pd.DataFrame) -> pd.Series:
        hgft = self._cache.get('hgft', None)
        if hgft is None:
            hgft = self._hgft(df=df)

        agft = self._cache.get('agft', None)
        if agft is None:
            agft = self._hgft(df=df)

        values = hgft - agft
        return values

    def _hagat(self, df: pd.DataFrame) -> pd.Series:
        hgat = self._cache.get('hgat', None)
        if hgat is None:
            hgat = self._hgft(df=df)

        agat = self._cache.get('agat', None)
        if agat is None:
            agat = self._agat(df=df)

        values = hgat - agat
        return values

    def _hwt(self, df: pd.DataFrame) -> pd.Series:
        total_wins = self._cache.get('total-wins', None)
        if total_wins is None:
            seasons = df['Season'].repeat(2).reset_index(drop=True)
            teams = df[['Home', 'Away']].stack().reset_index(drop=True)
            results = pd.DataFrame({'Home': df['Result'].eq('H'), 'Away': df['Result'].eq('A')}, dtype=int).stack().reset_index(drop=True)
            total_wins = self._cache['total-wins'] = results.groupby(by=[seasons, teams], sort=False, group_keys=False).transform(self._aggregate)

        values = total_wins.iloc[::2].reset_index(drop=True)
        return values

    def _hlt(self, df: pd.DataFrame) -> pd.Series:
        total_losses = self._cache.get('total-losses', None)
        if total_losses is None:
            seasons = df['Season'].repeat(2).reset_index(drop=True)
            teams = df[['Home', 'Away']].stack().reset_index(drop=True)
            results = pd.DataFrame({'Home': df['Result'].eq('A'), 'Away': df['Result'].eq('H')}, dtype=int).stack().reset_index(drop=True)
            total_losses = self._cache['total-losses'] = results.groupby(by=[seasons, teams], sort=False, group_keys=False).transform(self._aggregate)

        values = total_losses.iloc[::2].reset_index(drop=True)
        return values

    def _hgft(self, df: pd.DataFrame) -> pd.Series:
        total_goals = self._cache.get('total-goals-forward', None)
        if total_goals is None:
            seasons = df['Season'].repeat(2).reset_index(drop=True)
            teams = df[['Home', 'Away']].stack().reset_index(drop=True)
            goals = df[['HG', 'AG']].stack().reset_index(drop=True)
            total_goals = self._cache['total-goals-forward'] = goals.groupby(by=[seasons, teams], sort=False, group_keys=False).transform(self._aggregate)

        values = self._cache['hgft'] = total_goals.iloc[::2].reset_index(drop=True)
        return values

    def _hgat(self, df: pd.DataFrame) -> pd.Series:
        total_goals = self._cache.get('total-goals-against', None)
        if total_goals is None:
            seasons = df['Season'].repeat(2).reset_index(drop=True)
            teams = df[['Home', 'Away']].stack().reset_index(drop=True)
            goals = df[['AG', 'HG']].stack().reset_index(drop=True)
            total_goals = self._cache['total-goals-against'] = goals.groupby(by=[seasons, teams], sort=False, group_keys=False).transform(self._aggregate)

        values = self._cache['hgat'] = total_goals.iloc[::2].reset_index(drop=True)
        return values

    def _hwtp(self, df: pd.DataFrame) -> pd.Series:
        wtp = self._cache.get('wtp', None)
        if wtp is None:
            seasons = df['Season'].repeat(2).reset_index(drop=True)
            teams = df[['Home', 'Away']].stack().reset_index(drop=True)
            results = pd.DataFrame({'Home': df['Result'].eq('H'), 'Away': df['Result'].eq('A')}, dtype=int).stack().reset_index(drop=True)
            home_group = results.groupby(by=[seasons, teams], sort=False, group_keys=False)
            cumwins = home_group.cumsum() - results
            home_cc = home_group.cumcount()
            wtp = self._cache['wtp'] = (100*cumwins/home_cc).round(decimals=1)

        values = wtp.iloc[::2].reset_index(drop=True)
        return values

    def _hltp(self, df: pd.DataFrame) -> pd.Series:
        ltp = self._cache.get('ltp', None)
        if ltp is None:
            seasons = df['Season'].repeat(2).reset_index(drop=True)
            teams = df[['Home', 'Away']].stack().reset_index(drop=True)
            results = pd.DataFrame({'Home': df['Result'].eq('A'), 'Away': df['Result'].eq('H')}, dtype=int).stack().reset_index(drop=True)
            home_group = results.groupby(by=[seasons, teams], sort=False, group_keys=False)
            cumwins = home_group.cumsum() - results
            home_cc = home_group.cumcount()
            ltp = self._cache['ltp'] = (100*cumwins/home_cc).round(decimals=1)

        values = ltp.iloc[::2].reset_index(drop=True)
        return values

    def _awt(self, df: pd.DataFrame) -> pd.Series:
        total_wins = self._cache.get('total-wins', None)
        if total_wins is None:
            seasons = df['Season'].repeat(2).reset_index(drop=True)
            teams = df[['Home', 'Away']].stack().reset_index(drop=True)
            results = pd.DataFrame({'Home': df['Result'].eq('H'), 'Away': df['Result'].eq('A')}, dtype=int).stack().reset_index(drop=True)
            total_wins = self._cache['total-wins'] = results.groupby(by=[seasons, teams], sort=False, group_keys=False).transform(self._aggregate)

        values = total_wins.iloc[1::2].reset_index(drop=True)
        return values

    def _alt(self, df: pd.DataFrame) -> pd.Series:
        total_losses = self._cache.get('total-losses', None)
        if total_losses is None:
            seasons = df['Season'].repeat(2).reset_index(drop=True)
            teams = df[['Home', 'Away']].stack().reset_index(drop=True)
            results = pd.DataFrame({'Home': df['Result'].eq('A'), 'Away': df['Result'].eq('H')}, dtype=int).stack().reset_index(drop=True)
            total_losses = self._cache['total-losses'] = results.groupby(by=[seasons, teams], sort=False, group_keys=False).transform(self._aggregate)

        values = total_losses.iloc[1::2].reset_index(drop=True)
        return values

    def _agft(self, df: pd.DataFrame) -> pd.Series:
        total_goals = self._cache.get('total-goals-forward', None)
        if total_goals is None:
            seasons = df['Season'].repeat(2).reset_index(drop=True)
            teams = df[['Home', 'Away']].stack().reset_index(drop=True)
            goals = df[['HG', 'AG']].stack().reset_index(drop=True)
            total_goals = self._cache['total-goals-forward'] = goals.groupby(by=[seasons, teams], sort=False, group_keys=False).transform(self._aggregate)

        values = self._cache['agft'] = total_goals.iloc[1::2].reset_index(drop=True)
        return values

    def _agat(self, df: pd.DataFrame) -> pd.Series:
        total_goals = self._cache.get('total-goals-against', None)
        if total_goals is None:
            seasons = df['Season'].repeat(2).reset_index(drop=True)
            teams = df[['Home', 'Away']].stack().reset_index(drop=True)
            goals = df[['AG', 'HG']].stack().reset_index(drop=True)
            total_goals = self._cache['total-goals-against'] = goals.groupby(by=[seasons, teams], sort=False, group_keys=False).transform(self._aggregate)

        values = self._cache['agat'] = total_goals.iloc[1::2].reset_index(drop=True)
        return values

    def _awtp(self, df: pd.DataFrame) -> pd.Series:
        wtp = self._cache.get('wtp', None)
        if wtp is None:
            seasons = df['Season'].repeat(2).reset_index(drop=True)
            teams = df[['Home', 'Away']].stack().reset_index(drop=True)
            results = pd.DataFrame({'Home': df['Result'].eq('H'), 'Away': df['Result'].eq('A')}, dtype=int).stack().reset_index(drop=True)
            home_group = results.groupby(by=[seasons, teams], sort=False, group_keys=False)
            cumwins = home_group.cumsum() - results
            home_cc = home_group.cumcount()
            wtp = self._cache['wtp'] = (100*cumwins/home_cc).round(decimals=1)

        values = wtp.iloc[1::2].reset_index(drop=True)
        return values

    def _altp(self, df: pd.DataFrame) -> pd.Series:
        ltp = self._cache.get('ltp', None)
        if ltp is None:
            seasons = df['Season'].repeat(2).reset_index(drop=True)
            teams = df[['Home', 'Away']].stack().reset_index(drop=True)
            results = pd.DataFrame({'Home': df['Result'].eq('A'), 'Away': df['Result'].eq('H')}, dtype=int).stack().reset_index(drop=True)
            home_group = results.groupby(by=[seasons, teams], sort=False, group_keys=False)
            cumwins = home_group.cumsum() - results
            home_cc = home_group.cumcount()
            ltp = self._cache['ltp'] = (100*cumwins/home_cc).round(decimals=1)

        values = ltp.iloc[1::2].reset_index(drop=True)
        return values

    def compute_stats(self, df: pd.DataFrame, stats_columns: List[str]) -> pd.DataFrame:
        """ Computes the requested statistics for each season.

            Args
            ----
            :param df: (pd.DataFrame): League's donwloaded dataframe.
            :param stats_columns: (List[str]) List of statistics to compute.
        """

        def compute_stat(col: str) -> pd.Series:
            """ Helper function that calls the appropriate statistic computation function.

                Args
                ----
                :param col: (str) statistic to compute.
            """

            fn = self._stats_columns[col]
            values = fn(df=df).rename(col)
            return values

        # Validating operation.
        if not stats_columns:
            return df

        # Computing statistics.
        stats_dfs = [compute_stat(col=stat_col) for stat_col in tqdm(stats_columns, desc='Computing Statistics')]

        # Concatenating stats with league's dataframe.
        df = pd.concat([df] + stats_dfs, axis=1)

        # Clearing cache.
        self._cache.clear()
        return df
