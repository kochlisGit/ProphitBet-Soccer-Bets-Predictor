from analysis.features.importance import ImportanceAnalyzer
from tkinter import Toplevel, StringVar, messagebox
from tkinter.ttk import Combobox, Label
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class ImportancePlotter:
    def __init__(self, window, analyzer: ImportanceAnalyzer, show_help: bool):
        self._window = window
        self._analyzer = analyzer
        self._show_help = show_help

        self._colors = {color.name: color.value for color in analyzer.Colors}
        self._colors['Default'] = None
        self._methods = {
            'Importance Weights': analyzer.plot_feature_classification_weights,
            'Variance Analysis': analyzer.plot_feature_variances,
            'Univariate Test Importance': analyzer.plot_univariate_test_importance,
            'Feature Elimination Importance': analyzer.plot_feature_elimination_importance
        }

        self._title = 'Importance Plot'
        self._window_sizes = {'width': 1000, 'height': 800}
        self._paddings = {'padx': 10, 'pady': 5}

        self._window = None
        self._plotbar = None
        self._method_selection_var = None
        self._color_selection_var = None

        self._init_window()
        self._init_widgets()
        self._draw_plot()

        if self.show_help:
            messagebox.showinfo(
                'Feature Importance Info', 'Use "Importance Weights" method to train a model and plot its importance '
                                           'scores for each feature (column). Use "Variance Analysis to plot variance '
                                           'scores for each feature." Features with very low variances have little '
                                           'predictive power. Univariate test are statistical tests, which assign '
                                           'importance score to each feature. Finally, feature eliminator is a , '
                                           'process on which a model is trained multiple times, each time removing a '
                                           'feature. The importance scores are computed by the model\'s accuracy at '
                                           'each iteration.'
            )

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

    def _init_widgets(self):
        Label(self._window, text='Method').grid(row=0, column=0, **self._paddings)

        self._method_selection_var = StringVar()
        methods_cb = Combobox(
            self._window,
            width=30,
            values=list(self._methods.keys()),
            textvariable=self._method_selection_var,
            state='readonly'
        )
        methods_cb.current(0)
        methods_cb.bind("<<ComboboxSelected>>", self._update_plot)
        methods_cb.grid(row=1, column=0, **self._paddings)

        Label(self._window, text='Colors').grid(row=0, column=1, **self._paddings)

        self._color_selection_var = StringVar()
        color_list = list(self._colors.keys())
        color_list.reverse()
        color_cb = Combobox(
            self._window,
            values=color_list,
            textvariable=self._color_selection_var,
            state='readonly'
        )
        color_cb.current(0)
        color_cb.bind("<<ComboboxSelected>>", self._update_plot)
        color_cb.grid(row=1, column=1, **self._paddings)

    def _draw_plot(self):
        figure, ax = plt.subplots()
        plot_method = self._methods[self._method_selection_var.get()]
        plot_method(color=self._colors[self._color_selection_var.get()], ax=ax)

        self._plotbar = FigureCanvasTkAgg(figure, self._window)
        self._plotbar.get_tk_widget().grid(row=2, column=0, columnspan=3, sticky='nsew')

    def _update_plot(self, event=None):
        if self._plotbar is None:
            return

        self._plotbar.get_tk_widget().pack_forget()
        self._draw_plot()

    def open(self):
        self._window.mainloop()
