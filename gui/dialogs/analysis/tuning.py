import pandas as pd
from tkinter.ttk import Label
from models.tasks import ClassificationTask
from gui.dialogs.analysis.plotter import Plotter


class TuningImportancePlotter(Plotter):
    def __init__(self, root, task: ClassificationTask, importance_scores: dict[str, float]):
        super().__init__(root=root, title='Tuning Importance Analysis', window_size={'width': 900, 'height': 700})

        self._task = task
        self._importance_df = pd.DataFrame({
            'param': list(importance_scores.keys()),
            'importance': list(importance_scores.values())
        })

    def _create_widgets(self):
        Label(self._window, text=f'Task: {self._task}').grid(row=0, column=1, **self._paddings)

    def _init_dialog(self):
        self._plot(event=None)

    def _generate_plot(self, ax):
        self._importance_df.plot.bar(x='param', y='importance', ax=ax)
