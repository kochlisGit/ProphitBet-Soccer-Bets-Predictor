import numpy as np
from analysis.analyzer import FeatureAnalyzer


class CorrelationAnalyzer(FeatureAnalyzer):
    def __init__(self, results_and_stats, columns):
        super().__init__(results_and_stats, columns)
        self.correlation_columns = {
            'Home-Result': ['1', 'X', '2', 'HW', 'HL', 'HGF', 'HGD-W', 'HGD-L', 'HGD', 'HW%', 'HWD%'],
            'Away-Result': ['1', 'X', '2', 'AW', 'AL', 'AGF', 'AGD-W', 'AGD-L', 'AGD', 'AW%', 'AWD%']
        }

    def plot_feature_correlations(self, corr_columns, color_map, hide_upper_triangle, ax):
        d = self._inputs[corr_columns]
        corr = d.corr()
        if hide_upper_triangle:
            corr_mask = np.triu(np.ones_like(corr, dtype=bool))
        else:
            corr_mask = None
        self._plot_heatmap(data=corr, color_map=color_map, mask=corr_mask, ax=ax)
