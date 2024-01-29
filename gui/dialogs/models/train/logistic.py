import pandas as pd
from tkinter import StringVar
from tkinter.ttk import Combobox
from database.entities.leagues.league import LeagueConfig
from database.repositories.model import ModelRepository
from gui.dialogs.models.train.training import TrainingDialog
from models.estimators import LogisticRegression


class LogisticRegressionTrainDialog(TrainingDialog):
    def __init__(self, root, matches_df: pd.DataFrame, league_config: LeagueConfig, model_repository: ModelRepository):
        super().__init__(
            root=root,
            title='Logistic Regression Trainer',
            matches_df=matches_df,
            league_config=league_config,
            model_repository=model_repository
        )

        self._class_weights_list = ['None', 'balanced']
        self._penalty_list = ['l1', 'l2']

    def _get_model_cls(self):
        return LogisticRegression

    def _create_widgets(self):
        penalty_var = StringVar(value='l1')
        widget_params = {
            'master': self.window, 'values': self._penalty_list, 'state': 'readonly', 'font': ('Arial', 10), 'textvariable': penalty_var, 'width': 8
        }
        self._add_tunable_widget(
            key='penalty',
            widget_cls=Combobox,
            param_values=self._penalty_list,
            value_variable=penalty_var,
            name='Regularization',
            description='Weight Regularization (l1: l1-norm, l2: l2-norm)',
            x=360,
            y=250,
            x_pad=15,
            **widget_params
        )

        class_weight_var = StringVar(value='None')
        widget_params = {
            'master': self.window, 'values': self._class_weights_list, 'state': 'readonly', 'font': ('Arial', 10), 'textvariable': class_weight_var, 'width': 8
        }
        self._add_tunable_widget(
            key='class_weight',
            widget_cls=Combobox,
            param_values=self._class_weights_list,
            value_variable=class_weight_var,
            name='Class Weight',
            description='Assign higher learning importance to imbalanced class',
            x=360,
            y=350,
            x_pad=15,
            **widget_params
        )

    def _init_dialog(self):
        self._model_id_var.set(value='logistic-model')

        super()._init_dialog()

    def _get_dialog_result(self):
        return None
