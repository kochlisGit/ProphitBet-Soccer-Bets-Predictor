from analysis.features.analyzer import FeatureAnalyzer
import pandas as pd


class ClassDistributionAnalyzer(FeatureAnalyzer):
    def __init__(self, results_and_stats: pd.DataFrame):
        super().__init__(results_and_stats=results_and_stats)

    def plot_target_distribution(
            self,
            ax
    ):
        targets_counts = self.targets.value_counts()
        self._plot_bar(
            data=targets_counts.index, labels=targets_counts.values, color=None, ax=ax
        )
