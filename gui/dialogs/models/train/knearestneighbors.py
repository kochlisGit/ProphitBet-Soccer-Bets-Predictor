import pandas as pd
from tkinter import StringVar, IntVar, messagebox
from tkinter.ttk import Combobox
from database.entities.leagues.league import LeagueConfig
from database.repositories.model import ModelRepository
from gui.dialogs.models.train.training import TrainingDialog
from gui.widgets.intslider import IntSlider
from models.estimators import KNearestNeighbors


class KNearestNeighborsTrainDialog(TrainingDialog):
    def __init__(self, root, matches_df: pd.DataFrame, league_config: LeagueConfig, model_repository: ModelRepository):
        super().__init__(
            root=root,
            title='K-Nearest Neighbors Trainer',
            matches_df=matches_df,
            league_config=league_config,
            model_repository=model_repository
        )

        self._weights_list = ['uniform', 'distance']
        self._n_neighbors_range = (5, 251, 8)

    def _get_model_cls(self):
        return KNearestNeighbors

    def _create_widgets(self):
        n_neighbors_var = IntVar(value=5)
        widget_params = {
            'master': self.window, 'from_': self._n_neighbors_range[0], 'to': self._n_neighbors_range[1], 'variable': n_neighbors_var
        }
        self._add_tunable_widget(
            key='n_neighbors',
            widget_cls=IntSlider,
            param_values=self._n_neighbors_range,
            value_variable=n_neighbors_var,
            name='Neighbors',
            description='Number (K) of neighbors',
            x=360,
            y=250,
            x_pad=15,
            **widget_params
        )

        weights_var = StringVar(value='uniform')
        widget_params = {
            'master': self.window, 'values': self._weights_list, 'state': 'readonly', 'font': ('Arial', 10), 'textvariable': weights_var, 'width': 8
        }
        self._add_tunable_widget(
            key='weights',
            widget_cls=Combobox,
            param_values=self._weights_list,
            value_variable=weights_var,
            name='Neighbor Weights',
            description='Whether closer neighbors have greater influence on predictions or all neighbors are equal',
            x=360,
            y=350,
            x_pad=15,
            **widget_params
        )

    def _train(self):
        n_neighbors = self._tunable_widgets['n_neighbors'].get_value()

        if n_neighbors % 2 == 0:
            new_val = n_neighbors + 1
            self._tunable_widgets['n_neighbors'].set_value(value=new_val)

            messagebox.showwarning(
                parent=self.window,
                title='Incorrect Configuration',
                message=f'KNN requires an odd "K" neighbors number, so K was changed from {n_neighbors} to {new_val}'
            )

        super()._train()

    def _init_dialog(self):
        self._model_id_var.set(value='knn-model')

        super()._init_dialog()

    def _get_dialog_result(self):
        return None
