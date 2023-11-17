import pandas as pd
import seaborn
from analysis.features.analyzer import FeatureAnalyzer


class ClassDistributionAnalyzer(FeatureAnalyzer):
    def __init__(self, matches_df: pd.DataFrame):
        super().__init__(matches_df=matches_df)

    def plot(self, ax, **kwargs):
        targets = self.targets
        seaborn.barplot(x=["H", "D", "A"], y=targets.value_counts(), color=None, ax=ax)
