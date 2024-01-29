import pandas as pd
from gui.dialogs.analysis.plotter import Plotter
from analysis.variance import VarianceAnalyzer


class VariancePlotter(Plotter):
    def __init__(self, root, matches_df: pd.DataFrame):
        super().__init__(root=root, title='Variance Analysis', window_size={'width': 900, 'height': 700})

        self._analyzer = VarianceAnalyzer(df=matches_df)

    def _create_widgets(self):
        return

    def _init_dialog(self):
        self._plot(event=None)

    def _generate_plot(self, ax):
        self._analyzer.plot(ax=ax)
