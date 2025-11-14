import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional
from matplotlib.axes import Axes
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from src.analysis.analyzer import FeatureAnalyzer


class VarianceAnalyzer(FeatureAnalyzer):
    """ Variance analyzer, which measures the variance (spread) of each individual feature. """

    def __init__(self, df: pd.DataFrame):
        super().__init__(df=df)

    def _generate_plot(self, df: pd.DataFrame, colormap: Optional[str] = None, **kwargs) -> Axes:
        """ Applies Min-Max scaling to data and generates a variance analysis bar plot. """

        # Construct & Normalize inputs.
        x = df[self._trainable_features].to_numpy(dtype=np.float32)
        x_scaled = MinMaxScaler().fit_transform(x)

        # Analyze variance.
        feature_selector = VarianceThreshold()
        feature_selector.fit_transform(x_scaled)

        # Generate variances bar plot with descending order.
        variance_df = pd.DataFrame({
            'Variance': feature_selector.variances_,
            'Feature': self._trainable_features
        }).sort_values(by='Variance', ascending=False, ignore_index=True)
        _, ax = plt.subplots(constrained_layout=True)
        ax = sns.barplot(x=variance_df['Variance'], y=variance_df['Feature'], palette=colormap, edgecolor='black', ax=ax)
        ax.set_ylabel(None)
        ax.set_title('Normalized Variance Score per Feature')
        return ax
