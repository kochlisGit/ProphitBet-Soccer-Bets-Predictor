import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Literal, Optional
from matplotlib.axes import Axes
from src.analysis.analyzer import FeatureAnalyzer


class CorrelationAnalyzer(FeatureAnalyzer):
    """ Feature correlation analyzer, which generates the correlation heatmap."""

    def __init__(self, df: pd.DataFrame):
        super().__init__(df=df)

        self._methods = ['pearson', 'kendall', 'spearman']

    @property
    def methods(self) -> List[str]:
        return self._methods

    def _generate_plot(
            self,
            df: pd.DataFrame,
            method: Literal['pearson', 'kendall', 'spearman'] = 'pearson',
            colormap: Optional[str] = None,
            feature_type: Optional[str] = None
    ) -> Axes:
        """ Generates correlation heatmap between each feature pair. """

        input_df = df[self._trainable_features]

        # Select features.
        if feature_type is not None:
            if feature_type == 'home':
                input_df = input_df[['1', 'X', '2'] + [col for col in input_df.columns if col[0] == 'H']]
            elif feature_type == 'away':
                input_df = input_df[['1', 'X', '2'] + [col for col in input_df.columns if col[0] == 'A']]
            else:
                raise ValueError(f'Undefined feature type: "{feature_type}"')

        # Compute feature correlations.
        correlations = input_df.corr(method=method)

        # Generate heatmap plot.
        _, ax = plt.subplots(constrained_layout=True)
        mask = np.triu(np.ones_like(correlations, dtype=bool))
        ax = sns.heatmap(correlations, annot=True, cmap=colormap, mask=mask, ax=ax)
        ax.set_title('Feature Correlation Heatmap')
        return ax
