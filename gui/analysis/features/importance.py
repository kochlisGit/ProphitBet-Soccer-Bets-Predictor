import pandas as pd
from tkinter import StringVar
from tkinter.ttk import Label, Combobox
from analysis.features.importance import ImportanceAnalyzer
from gui.analysis.features.plotter import Plotter


class ImportancePlotter(Plotter):
    def __init__(self, root, matches_df: pd.DataFrame, show_help: bool):
        super().__init__(
            root=root,
            title='Feature Importance Plot',
            window_size={'width': 1000, 'height': 800},
            matches_df=matches_df,
            show_help=show_help
        )
        self._analyzer = ImportanceAnalyzer(matches_df=self.matches_df)

        self._methods = {
            'Variance Analysis': self._analyzer.plot_feature_variances,
            'Univariate Test Importance': self._analyzer.plot_univariate_test_importance,
            'Classifier Importance Weights': self._analyzer.plot_feature_classification_weights,
            'Feature Elimination Importance': self._analyzer.plot_feature_elimination_importance
        }
        self._method_selection_var = StringVar()

    def _get_help_message(self) -> str:
        return 'Use "Importance Weights" method to train a classification model and plot its importance ' \
               'scores for each feature (column). Use "Variance Analysis to plot variance ' \
               'scores for each feature." Features with very low variances have little ' \
               'predictive power, since the expected value of that feature is always similar. ' \
               'Univariate test are statistical tests, which assign an' \
               'importance score to each feature. Finally, feature elimination process is a , ' \
               'process on which a model is trained multiple times, each time removing a ' \
               'feature. The importance scores are computed by the model\'s accuracy at ' \
               'each iteration.'

    def _initialize(self):
        Label(self._window, text='Method').grid(row=0, column=0, **self._paddings)

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

    def _generate_plot(self, ax):
        method_name = self._method_selection_var.get()
        self._methods[method_name](ax=ax)
