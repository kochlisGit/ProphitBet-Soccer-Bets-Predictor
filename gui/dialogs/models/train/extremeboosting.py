import pandas as pd
from tkinter import IntVar, DoubleVar, Scale, messagebox
from database.entities.leagues.league import LeagueConfig
from database.repositories.model import ModelRepository
from gui.dialogs.models.train.training import TrainingDialog
from gui.widgets.intslider import IntSlider
from models.estimators import XGBoost


class ExtremeBoostingTrainDialog(TrainingDialog):
    def __init__(self, root, matches_df: pd.DataFrame, league_config: LeagueConfig, model_repository: ModelRepository):
        super().__init__(
            root=root,
            title='XGBoost Trainer',
            matches_df=matches_df,
            league_config=league_config,
            model_repository=model_repository
        )

        self._n_estimators_range = (50, 500, 25)
        self._learning_rate_range = (0.001, 0.3, 0.01)
        self._max_depth_range = (1, 10, 1)
        self._min_child_weight_range = (1, 7, 2)
        self._lambda_regularization_range = (0.0, 1.0, 0.25)
        self._alpha_regularization_range = (0.0, 1.0, 0.25)

    def _get_model_cls(self):
        return XGBoost

    def _init_dialog(self):
        self._model_id_var.set(value='xgboost-model')

        super()._init_dialog()

        self._tunable_widgets['calibrate_probabilities'].set_value(value=False)

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
            x=150,
            y=200,
            x_pad=15,
            **widget_params
        )

        lr_var = DoubleVar(value=1.0)
        widget_params = {
            'master': self.window, 'from_': self._learning_rate_range[0], 'to': self._learning_rate_range[1], 'tickinterval': 0.05, 'resolution': 0.01, 'orient': 'horizontal', 'length': 180, 'variable': lr_var
        }
        self._add_tunable_widget(
            key='learning_rate',
            widget_cls=Scale,
            param_values=self._learning_rate_range,
            value_variable=lr_var,
            name='Learning Rate',
            description='Learning rate of Gradient Descent',
            x=500,
            y=200,
            x_pad=15,
            **widget_params
        )

        max_depth_var = IntVar(value=6)
        widget_params = {
            'master': self.window, 'from_': self._max_depth_range[0], 'to': self._max_depth_range[1], 'variable': max_depth_var
        }
        self._add_tunable_widget(
            key='max_depth',
            widget_cls=IntSlider,
            param_values=self._max_depth_range,
            value_variable=max_depth_var,
            name='Max Depth',
            description='Max allowed decision tree depth',
            x=150,
            y=300,
            x_pad=15,
            **widget_params
        )

        child_weight_var = IntVar(value=1)
        widget_params = {
            'master': self.window, 'from_': self._min_child_weight_range[0], 'to': self._min_child_weight_range[1], 'variable': child_weight_var
        }
        self._add_tunable_widget(
            key='min_child_weight',
            widget_cls=IntSlider,
            param_values=self._min_child_weight_range,
            value_variable=child_weight_var,
            name='Min Child Weight',
            description='Min sum of instance weight required in a child node.',
            x=500,
            y=300,
            x_pad=15,
            **widget_params
        )

        lambda_reg_var = DoubleVar(value=1.0)
        widget_params = {
            'master': self.window, 'from_': self._lambda_regularization_range[0], 'to': self._lambda_regularization_range[1], 'tickinterval': 0.25, 'resolution': 0.1, 'orient': 'horizontal', 'length': 180, 'variable': lambda_reg_var
        }
        self._add_tunable_widget(
            key='lambda_regularization',
            widget_cls=Scale,
            param_values=self._lambda_regularization_range,
            value_variable=lambda_reg_var,
            name='l1 Penalty',
            description='l1 regularization term',
            x=150,
            y=400,
            x_pad=15,
            **widget_params
        )

        alpha_reg_var = DoubleVar(value=1.0)
        widget_params = {
            'master': self.window, 'from_': self._alpha_regularization_range[0], 'to': self._alpha_regularization_range[1], 'tickinterval': 0.25, 'resolution': 0.1, 'orient': 'horizontal', 'length': 180, 'variable': alpha_reg_var
        }
        self._add_tunable_widget(
            key='alpha_regularization',
            widget_cls=Scale,
            param_values=self._alpha_regularization_range,
            value_variable=alpha_reg_var,
            name='l2 Penalty',
            description='l2 regularization term',
            x=500,
            y=400,
            x_pad=15,
            **widget_params
        )

    def _train(self):
        calibrate_prob_widget = self._tunable_widgets['calibrate_probabilities']
        calibrate_prob_widget.uncheck()

        if calibrate_prob_widget.get_value():
            calibrate_prob_widget.set_value(value=False)
            messagebox.showwarning(
                parent=self.window,
                title='Incorrect Configuration',
                message=f'The outputs of XG-Boost cannot be calibrated. This option is set to False.'
            )

        super()._train()

    def _get_dialog_result(self):
        return None
