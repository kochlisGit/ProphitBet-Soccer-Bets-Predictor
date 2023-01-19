import matplotlib.pyplot as plt
import pandas as pd
from abc import ABC, abstractmethod
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Toplevel, messagebox


class Plotter(ABC):
    def __init__(self, root, title: str, window_size: dict, matches_df: pd.DataFrame, show_help: bool):
        self._root = root
        self._title = title
        self._window_size = window_size
        self._matches_df = matches_df

        window = Toplevel(self._root)
        window.title(self._title)
        window.geometry('{}x{}'.format(self._window_size['width'], self._window_size['height']))
        window.rowconfigure(2, weight=1)
        window.columnconfigure(1, weight=1)
        self._window = window
        self._paddings = {'padx': 10, 'pady': 5}
        self._plot_canvas = None

        if show_help:
            messagebox.showinfo('Hint', self._get_help_message())

    @property
    def window(self):
        return self._window

    @property
    def matches_df(self) -> pd.DataFrame:
        return self._matches_df

    @abstractmethod
    def _get_help_message(self) -> str:
        pass

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def _generate_plot(self, ax):
        pass

    def _plot(self):
        figure, ax = plt.subplots()
        self._generate_plot(ax=ax)
        self._plot_canvas = FigureCanvasTkAgg(figure, self._window)
        self._plot_canvas.get_tk_widget().grid(row=2, column=0, columnspan=3, sticky='nsew')

    def _update_plot(self, event=None):
        if self._plot_canvas is None:
            return

        self._plot_canvas.get_tk_widget().destroy()
        self._plot()

    def open(self):
        self._initialize()
        self._plot()
        self._window.mainloop()

    def close(self):
        self._window.destroy()
        self._window.quit()
