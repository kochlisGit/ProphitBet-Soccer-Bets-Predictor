from analysis.features.correlation import CorrelationAnalyzer
from tkinter import Toplevel, StringVar, IntVar, messagebox
from tkinter.ttk import Combobox, Checkbutton, Label
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class CorrelationPlotter:
    def __init__(self, window, analyzer: CorrelationAnalyzer, show_help: bool):
        self._window = window
        self._analyzer = analyzer
        self._show_help = show_help

        self._color_maps = {colormap.name: colormap.value for colormap in analyzer.ColorMaps}

        self._title = 'Correlations Plot'
        self._window_sizes = {'width': 1000, 'height': 800}
        self._paddings = {'padx': 10, 'pady': 5}

        self._window = None
        self._plotbar = None
        self._columns_selection_var = None
        self._colormap_selection_var = None
        self._hide_upper_triangle_var = None

        self._init_window()
        self._init_widgets()
        self._draw_plot()

        if self.show_help:
            messagebox.showinfo(
                'Correlation Info', 'Avoid using features (columns) with high correlation (close to 1.0). '
                                    'You can use one of the features instead. You may also use the feature  '
                                    'importance analysis to choose between two high correlated features. '
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
        Label(self._window, text='Columns').grid(row=0, column=0, **self._paddings)

        self._columns_selection_var = StringVar()
        column_cb = Combobox(
            self._window,
            values=list(self._analyzer.correlation_columns.keys()),
            textvariable=self._columns_selection_var,
            state='readonly'
        )
        column_cb.current(0)
        column_cb.bind("<<ComboboxSelected>>", self._update_plot)
        column_cb.grid(row=1, column=0, **self._paddings)

        Label(self._window, text='Colors').grid(row=0, column=1, **self._paddings)

        self._colormap_selection_var = StringVar()
        colormap_cb = Combobox(
            self._window,
            values=list(self._color_maps.keys()),
            textvariable=self._colormap_selection_var,
            state='readonly'
        )
        colormap_cb.current(0)
        colormap_cb.bind("<<ComboboxSelected>>", self._update_plot)
        colormap_cb.grid(row=1, column=1, **self._paddings)

        self._hide_upper_triangle_var = IntVar()
        hide_chkbtn = Checkbutton(
            self._window,
            text='Hide Upper Triangle',
            onvalue=1,
            offvalue=0,
            variable=self._hide_upper_triangle_var,
            command=self._update_plot
        )
        hide_chkbtn.grid(row=1, column=2, **self._paddings)

    def _draw_plot(self):
        figure, ax = plt.subplots()
        self._analyzer.plot_feature_correlations(
            corr_columns=self._analyzer.correlation_columns[self._columns_selection_var.get()],
            color_map=self._color_maps[self._colormap_selection_var.get()],
            hide_upper_triangle=self._hide_upper_triangle_var.get(),
            ax=ax
        )
        self._plotbar = FigureCanvasTkAgg(figure, self._window)
        self._plotbar.get_tk_widget().grid(row=2, column=0, columnspan=3, sticky='nsew')

    def _update_plot(self, event=None):
        if self._plotbar is None:
            return

        self._plotbar.get_tk_widget().pack_forget()
        self._draw_plot()

    def open(self):
        self._window.mainloop()
