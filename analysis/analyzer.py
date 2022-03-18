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

    def __init__(self, results_and_stats, columns):
        inputs_df = pd.DataFrame(results_and_stats, columns=columns)
        self._inputs, self.targets, self._labels = self._preprocess_data(inputs_df)

    @staticmethod
    def _preprocess_data(inputs_df):
        targets_df = inputs_df['Result']
        targets_df = targets_df.replace({'H': 1, 'D': 0, 'A': 2})
        inputs_df = inputs_df.drop(columns=['Date', 'Home Team', 'Away Team', 'HG', 'AG', 'Result'])
        labels = inputs_df.columns
        return inputs_df, targets_df.to_numpy(), labels

    @staticmethod
    def _plot_heatmap(data, color_map, mask, ax):
        seaborn.heatmap(data, annot=True, cmap=color_map, mask=mask, ax=ax)

    @staticmethod
    def _plot_bar(data, labels, color, ax):
        seaborn.barplot(x=data, y=labels, color=color, ax=ax)
