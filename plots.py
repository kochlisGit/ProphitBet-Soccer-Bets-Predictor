from analysis.features.correlation import CorrelationAnalyzer
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class CorrelationPlotter:
    def __init__(self, matches_df) -> None:
        self.matches_df = matches_df
        self._analyzer = CorrelationAnalyzer(matches_df=self.matches_df)


    def generate_plot(self) -> Figure:
        # columns = self._analyzer.home_columns if self._columns_to_plot_var.get() == 'Home Columns' else \
        #     self._analyzer.away_columns
        columns = self._analyzer.home_columns
        fig, ax = plt.subplots()
        ax = self._analyzer.plot(
            columns=columns,
            ax=ax
            # color_map=self._color_map_var.get(),
            # hide_upper_triangle=self._hide_upper_triangle_var.get(),
        )
        return fig