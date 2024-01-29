import numpy as np
import pandas as pd
import seaborn as sns
from analysis.analyzer import FeatureAnalyzer


class CorrelationAnalyzer(FeatureAnalyzer):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df=df)

        self._odds_columns = [col for col in ['1', 'X', '2'] if col in df.columns]
        self._home_columns = [col for col in self.columns if col[0] == 'H' and col != 'HG']
        self._away_columns = self._odds_columns + [col for col in self.columns if col[0] == 'A' and col != 'AG']
        self._all_columns = self._odds_columns + self._home_columns + self._away_columns
        self._team_columns = {
            'Home': self._odds_columns + self._home_columns,
            'Away': self._odds_columns + self._away_columns
        }
        self._correlations = {}

    @property
    def team_columns(self) -> dict[str, list[str]]:
        return self._team_columns

    @property
    def all_columns(self) -> list:
        return self._all_columns

    def plot(self, ax, team_column: str = 'Home', colormap: str = 'coolwarm', **kwargs):
        assert team_column == 'Home' or team_column == 'Away', f'Not defined team column: "{team_column}"'

        if team_column not in self._correlations:
            self._correlations[team_column] = self.input_df[self._team_columns[team_column]].corr()

        mask = np.triu(np.ones_like(self._correlations[team_column], dtype=bool))
        sns.heatmap(self._correlations[team_column], annot=True, cmap=colormap, mask=mask, ax=ax)
