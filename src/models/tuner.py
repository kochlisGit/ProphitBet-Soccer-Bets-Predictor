import optuna
import pandas as pd
from typing import Any, Dict, List, Type, Union
from src.models.trainer import Trainer


class Tuner:
    """ Classification Model tuner class. """

    def __init__(
           self,
            model_cls: Type,
            fixed_params: Dict[str, Any],
            tunable_params: Dict[str, Union[List, Dict[str, Any]]],
            df: pd.DataFrame,
            metric: str
    ):
        """
            :param model_cls: The model instance to be created.
            :param fixed_params: The fixed model parameters (remain frozen during the tuning process).
            :param tunable_params: The tunable model parameters (the hyper-parameters of the model that will be tuned).
            :param df: The dataset which will be used to tune the model parameters.
            :param metric: The evaluation metric to be optimized. Supports: Accuracy, F1, Precision, Recall.
        """

        if len(tunable_params) == 0:
            raise ValueError('tunable_params does not contain any tunable parameters. Tuning is aborted.')

        self._model_cls = model_cls
        self._fixed_params = fixed_params
        self._tunable_params = tunable_params
        self._metric = metric
        self._df = df

        self._trainer = Trainer()

    def tune(self, trials: int) -> optuna.Study:
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=trials, show_progress_bar=True)
        return study

    def _objective(self, trial: optuna.Trial) -> float:
        trial_params = self._tune_params(trial=trial)
        model_config = {**self._fixed_params, **trial_params}
        model = self._model_cls(**model_config)
        metrics_df = self._trainer.cross_validation(model=model, df=self._df)
        metric_score = metrics_df.loc[metrics_df['data'] == 'eval', self._metric].mean()
        return metric_score

    def _tune_params(self, trial: optuna.Trial) -> Dict[str, Union[List, Dict[str, Any]]]:
        trial_params = {}
        for param_name, param_values in self._tunable_params.items():
            if isinstance(param_values, list):
                suggested_val = trial.suggest_categorical(name=param_name, choices=param_values)
            else:
                if not isinstance(param_values, dict):
                    raise TypeError(f'Expected param values "{param_name}" to be list or dict, got {type(param_values)}.')

                low = param_values['low']
                high = param_values['high']
                step = param_values['step']

                if isinstance(low, int):
                    suggested_val = trial.suggest_int(name=param_name, low=low, high=high, step=step)
                else:
                    if not isinstance(low, float):
                        raise TypeError(f'Expected param values "{param_name}" to be int or float, got {type(low)}.')

                    suggested_val = trial.suggest_float(name=param_name, low=low, high=high, step=step)

            trial_params[param_name] = suggested_val
        return trial_params
