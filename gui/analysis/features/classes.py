import pandas as pd
from analysis.features.classes import ClassDistributionAnalyzer
from gui.analysis.features.plotter import Plotter


class ClassDistributionPlotter(Plotter):
    def __init__(self, root, matches_df: pd.DataFrame, show_help: bool):
        super().__init__(
            root=root,
            title='Class Distribution Plot',
            window_size={'width': 1000, 'height': 800},
            matches_df=matches_df,
            show_help=show_help
        )
        self._analyzer = ClassDistributionAnalyzer(matches_df=self.matches_df)

    def _get_help_message(self) -> str:
        return 'Class distribution should be uniform (Equal number of classes). ' \
               'Imbalanced classes might trick models to overestimate the majority ' \
               'class (e.g. Predict the class that occurs more often, which is usually 1). If a ' \
               'league contains imbalanced classes, use imbalanced-learning techniques, ' \
               'such as calibration, weighting, over-sampling & input noise'

    def _initialize(self):
        pass

    def _generate_plot(self, ax):
        self._analyzer.plot(ax=ax)
