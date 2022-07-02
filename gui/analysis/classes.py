from analysis.features.classes import ClassDistributionAnalyzer
from tkinter import Toplevel, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class ClassDistributionPlotter:
    def __init__(self, window, analyzer: ClassDistributionAnalyzer, show_help: bool):
        self._window = window
        self._analyzer = analyzer
        self._show_help = show_help

        self._color_maps = {colormap.name: colormap.value for colormap in analyzer.ColorMaps}

        self._title = 'Correlations Plot'
        self._window_sizes = {'width': 1000, 'height': 800}
        self._paddings = {'padx': 10, 'pady': 5}

        self._window = None
        self._plotbar = None

        self._init_window()
        self._draw_plot()

        if self.show_help:
            messagebox.showinfo('Class Distribution Info', 'Class distribution should be uniform. Imbalanced classes '
                                                           'might lead a model to overestimate the majority class or '
                                                           'underestimate the minority one. If a league contains '
                                                           'imbalanced class, use model calibration.')

    @property
    def show_help(self) -> bool:
        return self._show_help

    def _init_window(self):
        window = Toplevel(self._window)
        window.title(self._title)
        window.geometry('{}x{}'.format(self._window_sizes['width'], self._window_sizes['height']))
        window.rowconfigure(2, weight=1)
        window.columnconfigure(1, weight=1)
        self._window = window

    def _draw_plot(self):
        figure, ax = plt.subplots()
        self._analyzer.plot_target_distribution(
            ax=ax
        )
        self._plotbar = FigureCanvasTkAgg(figure, self._window)
        self._plotbar.get_tk_widget().grid(row=2, column=0, columnspan=3, sticky='nsew')

    def open(self):
        self._window.mainloop()
