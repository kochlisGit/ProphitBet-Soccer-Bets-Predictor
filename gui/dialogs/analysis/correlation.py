import pandas as pd
from tkinter import StringVar
from tkinter.ttk import Label, Combobox
from gui.dialogs.analysis.plotter import Plotter
from analysis.correlation import CorrelationAnalyzer


class CorrelationPlotter(Plotter):
    def __init__(self, root, matches_df: pd.DataFrame):
        super().__init__(root=root, title='Correlation Analysis', window_size={'width': 900, 'height': 700})

        self._analyzer = CorrelationAnalyzer(df=matches_df)
        self._colormap_dict = self._analyzer.colormap

        self._team_column_var = StringVar(value='Home')
        self._colormap_var = StringVar(value='Coolwarm')

    def _create_widgets(self):
        Label(self._window, text='Columns').grid(row=0, column=0, **self._paddings)
        column_cb = Combobox(
            self._window,
            values=list(self._analyzer.team_columns.keys()),
            textvariable=self._team_column_var,
            state='readonly'
        )
        column_cb.bind("<<ComboboxSelected>>", self._plot)
        column_cb.grid(row=0, column=1, **self._paddings)

        Label(self._window, text='Color').grid(row=1, column=0, **self._paddings)
        color_cb = Combobox(
            self._window,
            values=list(self._colormap_dict.keys()),
            textvariable=self._colormap_var,
            state='readonly'
        )
        color_cb.bind("<<ComboboxSelected>>", self._plot)
        color_cb.grid(row=1, column=1, **self._paddings)

    def _init_dialog(self):
        self._plot(event=None)

    def _generate_plot(self, ax):
        team_column = self._team_column_var.get()
        colormap = self._colormap_dict[self._colormap_var.get()]
        self._analyzer.plot(ax=ax, team_column=team_column, colormap=colormap)
