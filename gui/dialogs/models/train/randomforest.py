import pandas as pd
from tkinter import StringVar, IntVar
from tkinter.ttk import Combobox
from database.entities.leagues.league import LeagueConfig
from database.repositories.model import ModelRepository
from gui.dialogs.models.train.training import TrainingDialog
from gui.widgets.intslider import IntSlider
from models.estimators import RandomForest


class RandomForestTrainDialog(TrainingDialog):
    def __init__(self, root, matches_df: pd.DataFrame, league_config: LeagueConfig, model_repository: ModelRepository):
        super().__init__(
            root=root,
            title='Decision Tree Trainer',
            matches_df=matches_df,
            league_config=league_config,
            model_repository=model_repository
        )

        self._n_estimators_range = (50, 500, 25)
        self._criterion_list = ['gini', 'entropy', 'log_loss']
        self._min_samples_leaf_range = (1, 31, 2)
        self._min_samples_split_range = (2, 31, 2)
        self._max_depth_range = (0, 10, 1)
        self._class_weight_list = ['None', 'balanced']
        self._max_features_list = ['None', 'sqrt', 'log2']

    def _get_model_cls(self):
        return RandomForest

    def _init_dialog(self):
        self._model_id_var.set(value='random-forest-model')

        super()._init_dialog()

    def _create_widgets(self):
        n_estimators_var = IntVar(value=100)
        widget_params = {
            'master': self.window, 'from_': self._n_estimators_range[0], 'to': self._n_estimators_range[1], 'variable': n_estimators_var
        }
        self._add_tunable_widget(
            key='n_estimators',
            widget_cls=IntSlider,
            param_values=self._n_estimators_range,
            value_variable=n_estimators_var,
            name='Estimators',
            description='Number of decision trees (estimators)',
            x=360,
            y=190,
            x_pad=15,
            **widget_params
        )

        criterion_var = StringVar(value='gini')
        widget_params = {
            'master': self.window, 'values': self._criterion_list, 'state': 'readonly', 'font': ('Arial', 10), 'textvariable': criterion_var, 'width': 10
        }
        self._add_tunable_widget(
            key='criterion',
            widget_cls=Combobox,
            param_values=self._criterion_list,
            value_variable=criterion_var,
            name='Criterion',
            description='Criterion (loss metric) for constructing the tree nodes',
            x=185,
            y=270,
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
            x=185,
            y=350,
            x_pad=15,
            **widget_params
        )

        max_features_var = StringVar(value='sqrt')
        widget_params = {
            'master': self.window, 'values': self._max_features_list, 'state': 'readonly', 'font': ('Arial', 10), 'textvariable': max_features_var, 'width': 8
        }
        self._add_tunable_widget(
            key='max_features',
            widget_cls=Combobox,
            param_values=self._max_features_list,
            value_variable=max_features_var,
            name='Max Features',
            description='Max selected features for tree construction. In comparison to Decision Trees, it is best to avoid None.',
            x=185,
            y=430,
            x_pad=15,
            **widget_params
        )

        samples_leaf_var = IntVar(value=1)
        widget_params = {
            'master': self.window, 'from_': self._min_samples_leaf_range[0], 'to': self._min_samples_leaf_range[1], 'variable': samples_leaf_var
        }
        self._add_tunable_widget(
            key='min_samples_leaf',
            widget_cls=IntSlider,
            param_values=self._min_samples_leaf_range,
            value_variable=samples_leaf_var,
            name='Min Samples Leaf',
            description='Min required samples to form a leaf node (terminal node)',
            x=510,
            y=270,
            x_pad=15,
            **widget_params
        )

        max_depth_var = IntVar(value=0)
        widget_params = {
            'master': self.window, 'from_': self._max_depth_range[0], 'to': self._max_depth_range[1], 'variable': max_depth_var
        }
        self._add_tunable_widget(
            key='max_depth',
            widget_cls=IntSlider,
            param_values=self._max_depth_range,
            value_variable=max_depth_var,
            name='Max Depth',
            description='Max allowed tree depth. Select 0 to auto-adjust.',
            x=510,
            y=350,
            x_pad=15,
            **widget_params
        )

        samples_split_var = IntVar(value=1)
        widget_params = {
            'master': self.window, 'from_': self._min_samples_split_range[0], 'to': self._min_samples_split_range[1], 'variable': samples_split_var
        }
        self._add_tunable_widget(
            key='min_samples_split',
            widget_cls=IntSlider,
            param_values=self._min_samples_split_range,
            value_variable=samples_split_var,
            name='Min Samples Split',
            description='Min required samples to split a tree node',
            x=510,
            y=430,
            x_pad=15,
            **widget_params
        )

    def _get_dialog_result(self):
        return None
