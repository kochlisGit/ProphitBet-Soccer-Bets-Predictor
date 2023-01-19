from analysis.features.analyzer import FeatureAnalyzer
import numpy as np
import seaborn
import pandas as pd


class CorrelationAnalyzer(FeatureAnalyzer):
    def __init__(self, matches_df: pd.DataFrame):
        super().__init__(matches_df=matches_df)

        self._home_columns = [home_col for home_col in self.inputs.columns if home_col[0] != 'A']
        self._away_columns = [home_col for home_col in self.inputs.columns if home_col[0] != 'H']

    @property
    def home_columns(self) -> list:
        return self._home_columns

    @property
    def away_columns(self) -> list:
        return self._away_columns

    def plot(self, columns: np.ndarray or list, color_map: str, hide_upper_triangle: bool, ax, **kwargs):
        d = self._inputs[columns]
        correlations = d.corr()
        mask = np.triu(np.ones_like(correlations, dtype=bool)) if hide_upper_triangle else None
        seaborn.heatmap(correlations, annot=True, cmap=color_map, mask=mask, ax=ax)
