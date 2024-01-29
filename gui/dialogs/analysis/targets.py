import pandas as pd
from tkinter import StringVar
from tkinter.ttk import Label, Combobox
from models.tasks import ClassificationTask
from gui.dialogs.analysis.plotter import Plotter
from analysis.targets import TargetAnalyzer


class TargetPlotter(Plotter):
    def __init__(self, root, matches_df: pd.DataFrame):
        super().__init__(root=root, title='Target Distribution Analysis', window_size={'width': 900, 'height': 700})

        self._analyzer = TargetAnalyzer(df=matches_df)

        self._tasks = {
            ClassificationTask.Result.value: ClassificationTask.Result,
            ClassificationTask.Over.value: ClassificationTask.Over
        }
        self._task_var = StringVar(value='Result')

    def _create_widgets(self):
        Label(self._window, text='Tasks').grid(row=0, column=0, **self._paddings)

        column_cb = Combobox(
            self._window,
            values=list(self._tasks.keys()),
            textvariable=self._task_var,
            state='readonly'
        )
        column_cb.bind("<<ComboboxSelected>>", self._plot)
        column_cb.grid(row=0, column=1, **self._paddings)

    def _init_dialog(self):
        self._plot(event=None)

    def _generate_plot(self, ax):
        task = self._tasks[self._task_var.get()]
        self._analyzer.plot(ax=ax, task=task)
