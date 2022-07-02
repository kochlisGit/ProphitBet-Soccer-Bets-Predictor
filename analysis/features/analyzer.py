import numpy as np
import pandas as pd
import seaborn
from enum import Enum
from abc import ABC


class FeatureAnalyzer(ABC):
    class ColorMaps(Enum):
        Coolwarm = 'coolwarm'
        Rocket = 'rocket'
        Icefire = 'icefire'
        Crest = 'crest'
        Blues = 'Blues'

    class Colors(Enum):
        Blue = 'blue'
        Cyan = 'cyan'
        Red = 'red'
        Orange = 'orange'
        Green = 'green'

    def __init__(self, results_and_stats: pd.DataFrame):
        self._inputs, self._targets, self._labels = self._preprocess_data(results_and_stats)

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def labels(self):
        return self._labels

    @staticmethod
    def _preprocess_data(inputs_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, list):
        targets_df = inputs_df['Result'].replace({'H': 0, 'D': 1, 'A': 2})
        inputs_df = inputs_df.drop(columns=['Date', 'Home Team', 'Away Team', 'HG', 'AG', 'Result'])
        return inputs_df, targets_df, inputs_df.columns

    @staticmethod
    def _plot_heatmap(data: np.ndarray or pd.DataFrame, color_map: str, mask: np.ndarray, ax):
        seaborn.heatmap(data, annot=True, cmap=color_map, mask=mask, ax=ax)

    @staticmethod
    def _plot_bar(data: np.ndarray or pd.DataFrame, labels: list, color: str or None, ax):
        seaborn.barplot(x=data, y=labels, color=color, ax=ax)
