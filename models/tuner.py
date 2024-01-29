import optuna
import optuna.importance
import pandas as pd
from optuna.trial import Trial
from models.tasks import ClassificationTask
from models.trainer import Trainer


class Tuner:
    def __init__(self, trainer: Trainer):
        self._trainer = trainer
        self._importance_evaluator = optuna.importance.FanovaImportanceEvaluator(seed=0)

    @staticmethod
    def _suggest_neural_network_fc_hiddens(trial, values: tuple[int]) -> list[int]:
        hidden_layers = trial.suggest_int(name='hidden_layers', low=1, high=5)

        return [
            trial.suggest_int(name=f'layer_{i}', low=values[0], high=values[1], step=values[2])
            for i in range(1, hidden_layers + 1)
        ]

    def _get_trial_params(self, trial: Trial, tune_params: dict) -> dict:
        def suggest(param_name: str, values):
            if param_name == 'fc_hiddens':
                return self._suggest_neural_network_fc_hiddens(trial=trial, values=values)

            if isinstance(values, list):
                return trial.suggest_categorical(name=param_name, choices=values)
            else:
                assert isinstance(values, tuple) and len(values) == 3, \
                    f'Expected tuple with min value, max value, step, got {values}'

                if isinstance(values[0], int):
                    assert isinstance(values[1], int), f'Provided different types of min, max values: min = {type(values[0])} vs max = {type(values[1])}'

                    return trial.suggest_int(name=param_name, low=values[0], high=values[1], step=values[2])
                elif isinstance(values[1], float):
                    assert isinstance(values[1], float), f'Provided different types of min, max values: min = {type(values[0])} vs max = {type(values[1])}'

                    return trial.suggest_float(name=param_name, low=values[0], high=values[1], step=values[2])
                else:
                    raise NotImplementedError(f'Only int or float values are supported for range search, got min = {type(values[0])} vs max = {type(values[1])}')

        return {
            param_name: suggest(param_name=param_name, values=values)
            for param_name, values in tune_params.items()
        }

    def tune(
            self,
            n_trials: int,
            metric: str,
            df: pd.DataFrame,
            league_id: str,
            model_id: str,
            task: ClassificationTask,
            model_cls: type,
            model_params: dict[str, int or float or bool or None],
            tune_params: dict[str, tuple or list]
    ) -> optuna.Study:
        def objective(trial):
            trial_params = self._get_trial_params(trial=trial, tune_params=tune_params)
            all_params = {**model_params, **trial_params}

            cv_eval_dict = self._trainer.cross_validate(
                df=df,
                league_id=league_id,
                model_id=model_id,
                task=task,
                model_cls=model_cls,
                model_params=all_params
            )
            return cv_eval_dict[metric]

        assert n_trials > 0, f'n_trials should be a positive integer, got {n_trials}.'
        assert len(tune_params) > 0, 'At least 1 tunable param is required, got 0.'

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return study

    def get_param_importance_scores(self, study: optuna.Study) -> dict[str, float]:
        return optuna.importance.get_param_importances(study=study, evaluator=self._importance_evaluator)
