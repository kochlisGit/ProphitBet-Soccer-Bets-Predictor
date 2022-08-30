from analysis.features.analyzer import FeatureAnalyzer
import numpy as np
import pandas as pd


class CorrelationAnalyzer(FeatureAnalyzer):
    def __init__(self, results_and_stats: pd.DataFrame):
        super().__init__(results_and_stats=results_and_stats)
        self.correlation_columns = {
            'Home-Result': ['1', 'X', '2', 'HW', 'HL', 'HGF', 'HGA', 'HGD-W', 'HGD-L', 'HW%', 'HD%'],
            'Away-Result': ['1', 'X', '2', 'AW', 'AL', 'AGF', 'AGA', 'AGD-W', 'AGD-L', 'AW%', 'AD%']
        }

    def plot_feature_correlations(
            self,
            corr_columns: list,
            color_map: str,
            hide_upper_triangle: np.ndarray,
            ax
    ):
        d = self._inputs[corr_columns]
        corr = d.corr()
        if hide_upper_triangle:
            corr_mask = np.triu(np.ones_like(corr, dtype=bool))
        else:
            corr_mask = None
        self._plot_heatmap(data=corr, color_map=color_map, mask=corr_mask, ax=ax)
