import pandas as pd
from tkinter import StringVar, messagebox
from tkinter.ttk import Combobox
from database.entities.leagues.league import LeagueConfig
from database.repositories.model import ModelRepository
from gui.dialogs.models.train.training import TrainingDialog
from models.estimators import NaiveBayes


class NaiveBayesTrainDialog(TrainingDialog):
    def __init__(self, root, matches_df: pd.DataFrame, league_config: LeagueConfig, model_repository: ModelRepository):
        super().__init__(
            root=root,
            title='Naive Bayes Trainer',
            matches_df=matches_df,
            league_config=league_config,
            model_repository=model_repository
        )

        self._algorithm_list = ['gaussian', 'multinomial', 'complement']

    def _get_model_cls(self):
        return NaiveBayes

    def _init_dialog(self):
        self._model_id_var.set(value='bayes-model')

        super()._init_dialog()

    def _create_widgets(self):
        algorithm_var = StringVar(value='gaussian')
        widget_params = {
            'master': self.window, 'values': self._algorithm_list, 'state': 'readonly', 'font': ('Arial', 10), 'textvariable': algorithm_var, 'width': 12
        }
        self._add_tunable_widget(
            key='algorithm',
            widget_cls=Combobox,
            param_values=self._algorithm_list,
            value_variable=algorithm_var,
            name='Algorithm',
            description='Algorithm based on data distribution. Complement is best-suited for imbalanced targets.',
            x=340,
            y=280,
            x_pad=15,
            **widget_params
        )

    def _tune_model(self, tune_params: dict, model_params: dict) -> (dict, bool):
        if 'algorithm' in tune_params and 'normalizer' in tune_params:
            del tune_params['normalizer']

        model_params['normalizer'] = 'Min-Max'
        return super()._tune_model(tune_params=tune_params, model_params=model_params)

    def _train(self):
        algorithm = self._tunable_widgets['algorithm'].get_value()
        if algorithm != 'gaussian' and self._tunable_widgets['normalizer'].get_value() != 'Min-Max':
            messagebox.showwarning(
                parent=self.window,
                title='Incorrect Configuration',
                message=f'{algorithm} Naive Bayes should use Min-Max scaler. Switched to Min-Max scaler'
            )
            self._tunable_widgets['normalizer'].set_value(value='Min-Max')

        super()._train()

    def _get_dialog_result(self):
        return None
