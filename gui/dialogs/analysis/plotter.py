import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from gui.dialogs.dialog import Dialog


class Plotter(Dialog, ABC):
    def __init__(self, root, title: str, window_size: dict):
        super().__init__(root=root, title=title, window_size=window_size)

        self._window.rowconfigure(2, weight=1)
        self._window.columnconfigure(1, weight=1)
        self._paddings = {'padx': 10, 'pady': 5}

        self._plot_canvas = None

    @abstractmethod
    def _create_widgets(self):
        pass

    @abstractmethod
    def _init_dialog(self):
        pass

    @abstractmethod
    def _generate_plot(self, ax):
        pass

    def _get_dialog_result(self):
        return

    def _plot(self, event):
        if self._plot_canvas is not None:
            self._plot_canvas.get_tk_widget().destroy()

        figure, ax = plt.subplots()
        self._plot_canvas = FigureCanvasTkAgg(figure, self._window)
        self._plot_canvas.get_tk_widget().grid(row=2, column=0, columnspan=3, sticky='nsew')
        self._generate_plot(ax=ax)
