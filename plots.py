from analysis.features.correlation import CorrelationAnalyzer
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import base64
import io


class CorrelationPlotter:
    def __init__(self, matches_df) -> None:
        self.matches_df = matches_df
        self._analyzer = CorrelationAnalyzer(matches_df=self.matches_df)


    def generate_image(self) -> Figure:
        # columns = self._analyzer.home_columns if self._columns_to_plot_var.get() == 'Home Columns' else \
        #     self._analyzer.away_columns
        columns = self._analyzer.home_columns
        fig, ax = plt.subplots(figsize=(15, 15))
        ax = self._analyzer.plot(
            columns=columns,
            ax=ax
            # color_map=self._color_map_var.get(),
            # hide_upper_triangle=self._hide_upper_triangle_var.get(),
        )
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        output.seek(0)
        encoded_image = base64.b64encode(output.read()).decode('utf-8')
        return encoded_image