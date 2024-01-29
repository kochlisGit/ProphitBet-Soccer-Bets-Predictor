import pandas as pd
from tkinter import StringVar, DoubleVar, Scale
from tkinter.ttk import Combobox
from database.entities.leagues.league import LeagueConfig
from database.repositories.model import ModelRepository
from gui.dialogs.models.train.training import TrainingDialog
from models.estimators import SupportVectorMachine


class SupportVectorMachineTrainDialog(TrainingDialog):
    def __init__(self, root, matches_df: pd.DataFrame, league_config: LeagueConfig, model_repository: ModelRepository):
        super().__init__(
            root=root,
            title='SVM Trainer',
            matches_df=matches_df,
            league_config=league_config,
            model_repository=model_repository
        )

        self._C_range = (0.25, 2.0, 0.25)
        self._kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
        self._gamma_list = ['scale', 'auto']
        self._class_weight_list = ['None', 'balanced']

    def _get_model_cls(self):
        return SupportVectorMachine

    def _init_dialog(self):
        self._model_id_var.set(value='svm-model')

        super()._init_dialog()

    def _create_widgets(self):
        C_var = DoubleVar(value=1.0)
        widget_params = {
            'master': self.window, 'from_': self._C_range[0], 'to': self._C_range[1], 'tickinterval': 0.5, 'resolution': 0.1, 'orient': 'horizontal', 'length': 150, 'variable': C_var
        }
        self._add_tunable_widget(
            key='C',
            widget_cls=Scale,
            param_values=self._C_range,
            value_variable=C_var,
            name='Regularization Coeff',
            description='Regularization Coefficient',
            x=150,
            y=250,
            x_pad=15,
            **widget_params
        )

        kernel_var = StringVar(value='linear')
        widget_params = {
            'master': self.window, 'values': self._kernel_list, 'state': 'readonly', 'font': ('Arial', 10), 'textvariable': kernel_var, 'width': 8
        }
        self._add_tunable_widget(
            key='kernel',
            widget_cls=Combobox,
            param_values=self._kernel_list,
            value_variable=kernel_var,
            name='Kernel',
            description='SVM Kernel function',
            x=575,
            y=250,
            x_pad=15,
            **widget_params
        )

        gamma_var = StringVar(value='scale')
        widget_params = {
            'master': self.window, 'values': self._gamma_list, 'state': 'readonly', 'font': ('Arial', 10), 'textvariable': gamma_var, 'width': 8
        }
        self._add_tunable_widget(
            key='gamma',
            widget_cls=Combobox,
            param_values=self._gamma_list,
            value_variable=gamma_var,
            name='Gamma',
            description='Kernel Coefficient for rbf, poly, sigmoid',
            x=150,
            y=340,
            x_pad=15,
            **widget_params
        )

        class_weight_var = StringVar(value='None')
        widget_params = {
            'master': self.window, 'values': self._class_weight_list, 'state': 'readonly', 'font': ('Arial', 10), 'textvariable': class_weight_var, 'width': 8
        }
        self._add_tunable_widget(
            key='class_weight',
            widget_cls=Combobox,
            param_values=self._class_weight_list,
            value_variable=class_weight_var,
            name='Class Weight',
            description='Assign higher learning importance to imbalanced class',
            x=550,
            y=340,
            x_pad=15,
            **widget_params
        )

    def _get_dialog_result(self):
        return None
