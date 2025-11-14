import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Optional
from matplotlib.axes import Axes
from src.analysis.analyzer import FeatureAnalyzer


class DistributionAnalyzer(FeatureAnalyzer):
    """ Distribution analyzer for each individual feature. """

    def __init__(self, df: pd.DataFrame):
        super().__init__(df=df)

        self._all_features = df.columns.drop(['Date', 'Home', 'Away']).tolist()

    @property
    def all_features(self) -> List[str]:
        return self._all_features

    def _generate_plot(self, df: pd.DataFrame, colormap: Optional[str] = None, column: str = None) -> Axes:
        """ Generates distribution plot (histogram) for a selected column. """

        var = df[column]
        _, ax = plt.subplots(constrained_layout=True)

        if var.dtype == object:
            value_counts = var.value_counts(sort=False)
            ax = sns.barplot(x=value_counts.index.values, y=value_counts.values, palette=colormap, edgecolor='black')
            ax.set_title(f'Bar Plot of {column}')
        else:
            ax = sns.histplot(data=df, x=column, kde=True, palette=colormap, edgecolor='black')
            ax.set_title(f'Histogram Plot of {column}')

        ax.set_xlabel(column)
        ax.set_ylabel('Counts')
        return ax
