import pandas as pd
from tkinter import StringVar, BooleanVar
from tkinter.ttk import Label, Combobox, Checkbutton
from analysis.features.correlation import CorrelationAnalyzer
from gui.analysis.features.plotter import Plotter


class CorrelationPlotter(Plotter):
    def __init__(self, root, matches_df: pd.DataFrame, show_help: bool):
        super().__init__(
            root=root,
            title='Feature Correlation Plot',
            window_size={'width': 1000, 'height': 800},
            matches_df=matches_df,
            show_help=show_help
        )
        self._analyzer = CorrelationAnalyzer(matches_df=self.matches_df)

        self._columns_to_plot_var = StringVar()
        self._color_map_var = StringVar()
        self._hide_upper_triangle_var = BooleanVar(value=True)

    def _get_help_message(self) -> str:
        return 'Avoid using features (columns) with high correlation (close to 1.0). ' \
               'If there are 2 columns that are highly correlated (e.g. correlation ' \
               'score is close to 1.0 or -1.0, then it is best to exclude one of the columns'

    def _initialize(self):
        Label(self._window, text='Columns').grid(row=0, column=0, **self._paddings)
        Label(self._window, text='Colors').grid(row=0, column=1, **self._paddings)

        column_cb = Combobox(
            self._window,
            values=['Home Columns', 'Away Columns'],
            textvariable=self._columns_to_plot_var,
            state='readonly'
        )
        column_cb.current(0)
        column_cb.bind("<<ComboboxSelected>>", self._update_plot)
        column_cb.grid(row=1, column=0, **self._paddings)

        colormap_cb = Combobox(
            self._window,
            values=[color_map.value for color_map in self._analyzer.ColorMaps],
            textvariable=self._color_map_var,
            state='readonly'
        )
        colormap_cb.current(0)
        colormap_cb.bind("<<ComboboxSelected>>", self._update_plot)
        colormap_cb.grid(row=1, column=1, **self._paddings)

        hide_triangle_checkbutton = Checkbutton(
            self._window,
            text='Hide Upper Triangle',
            onvalue=True,
            offvalue=False,
            variable=self._hide_upper_triangle_var,
            command=self._update_plot
        )
        hide_triangle_checkbutton.grid(row=1, column=2, **self._paddings)

    def _generate_plot(self, ax):
        columns = self._analyzer.home_columns if self._columns_to_plot_var.get() == 'Home Columns' else \
            self._analyzer.away_columns
        self._analyzer.plot(
            columns=columns,
            color_map=self._color_map_var.get(),
            hide_upper_triangle=self._hide_upper_triangle_var.get(),
            ax=ax
        )
