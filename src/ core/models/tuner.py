import threading
import optuna
import pandas as pd
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from src.core.models.dashboard import start_optuna_dashboard, stop_optuna_dashboard, wait_optuna_server
from src.core.models.trainer import Trainer
from src.core.preprocessing.utils import construct_targets
from src.core.preprocessing.utils.selection import cross_validation


class TuningAlgorithm(Enum):
    """ The supported tuning algorithms.

        Supported Algorithms:
            1. TPE (Tree-structure Parzen Estimator Method).
            2. GP (Gaussian Process Bayesian Optimization Method).
    """

    TPE = 'tpe'
    GP = 'gp'

    def create(self, seed: Optional[int] = 0) -> optuna.samplers.BaseSampler:
        """ Factory method that creates a new optuna tuning algorithm. """

        if self == TuningAlgorithm.TPE:
            return optuna.samplers.TPESampler(seed=seed)
        elif self == TuningAlgorithm.GP:
            return optuna.samplers.GPSampler(seed=seed)
        else:
            raise KeyError(f'Not supported tuning algorithm: "{self.name}"')


class OptimizationMetric(Enum):
    """ Supported evaluation metrics that can be optimized.

        Supported Metrics:
            1. Accuracy
            2. F1
            3. Log Loss
            4. Ranked Probability Scores (RPS)
            5. PRECISION
            6. RECALL
    """

    ACC = ('Accuracy', 'maximize')
    F1 = ('F1', 'maximize')
    LOG_LOSS = ('Log Loss', 'minimize')
    RPS = ('RPS', 'minimize')
    PREC = ('Precision', 'maximize')
    REC = ('Recall', 'maximize')


class Tuner:
    """ Classification Model tuner class.

        Attributes
        ------
        :param model_cls: (Type[Classifier]) classifier's instance to be created.
        :param model_params: fixed model parameters (remain frozen during the tuning process).
        :param tunable_params: tunable model parameters (the hyperparameters of the model that will be tuned).
        :param metric: evaluation metric to be optimized. Supports: Accuracy, F1, Log Loss, RPS
    """

    def __init__(
            self,
            model_cls: Type,
            model_params: Dict[str, Any],
            tunable_params: Dict[str, Union[List, Dict[str, Any]]],
            metric: OptimizationMetric
    ):
        if len(tunable_params) == 0:
            raise ValueError('tunable_params does not contain any tunable parameters. Tuning is aborted.')

        self._model_cls = model_cls
        self._model_params = model_params
        self._tunable_params = tunable_params
        self._metric, self._direction = metric.value

        self._trainer = Trainer()

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Union[List, Dict[str, Any]]]:
        """ Suggests/Samples parameters using Optuna's trial module.

            Args
            ----
            :param trial: (optuna.trial.Trial) Optuna's trial object used to sample parameters.
        """

        trial_params = {}
        for param_name, param_values in self._tunable_params.items():
            if isinstance(param_values, list):
                suggested_val = trial.suggest_categorical(name=param_name, choices=param_values)
            else:
                if not isinstance(param_values, dict):
                    raise TypeError(
                        f'Expected param values "{param_name}" to be list or dict, got {type(param_values)}.'
                    )

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

    def _objective(self, df: pd.DataFrame, seed: Optional[int]) -> Callable:
        """ Optuna's objective function. It uses k-fold cross validation to optimize the classifiers.

            Args
            ----
            :param df: (pd.DataFrame) dataset to be used for optimization.
            :param seed: (bool) cross-validation random seed.
        """

        def objective(trial: optuna.Trial) -> float:
            trial_params = self._suggest_params(trial=trial)
            model_config = {**self._model_params, **trial_params}
            metrics_df = self._trainer.cross_validation(
                df=df,
                model_cls=self._model_cls,
                model_config=model_config,
                show_progress=False,
                seed=seed
            )
            return metrics_df.loc[metrics_df['Data'] == 'Eval', self._metric].mean()
        return objective

    def _fast_objective(self, df: pd.DataFrame, seed: Optional[int]) -> Callable:
        """ Optuna's fast objective function. It uses stratified holdout validation to optimize classifiers.
            It is faster than k-fold cross validation, but less reliable.

            Subsets:
                * Train: 80%
                * Eval: 20%

            Args
            ----
            :param df: (pd.DataFrame) dataset to be used for optimization.
            :param seed: (bool) cross-validation random seed.
        """

        def objective(trial: optuna.Trial) -> float:
            trial_params = self._suggest_params(trial=trial)
            model_config = {**self._model_params, **trial_params}
            model = self._model_cls(**model_config)
            _, metrics_df = self._trainer.train(model=model, df=df, test_size=0.2, seed=seed)
            return metrics_df.loc[metrics_df['Data'] == 'Eval', self._metric].mean()
        return objective

    def tune(
            self,
            df: pd.DataFrame,
            trials: int,
            algorithm: TuningAlgorithm,
            return_importances: bool = True,
            fast_mode: bool = False,
            storage_url: Optional[str] = None,
            study_name: Optional[str] = None,
            seed: Optional[int] = 0,
            open_optuna_dashboard: bool = False
    ) -> Tuple[pd.DataFrame, float, Dict[str, Any], Optional[Dict[str, float]]]:
        """ Tunes the model using the cross validation method and returns the trial dataframe. Returns the tune study,
            the evaluation error of the specified metric, the best combination of hyperparameters,
            and optionally the hyperparameter importances.

            Args
            ----
            :param df (pd.DataFrame) dataset to be used for tuning.
            :param trials: (int) number of trials for the hyperparameter search process.
            :param algorithm: (TuningAlgorithm) sampling algorithm to be used for tuning.
            :param return_importances: (bool) whether to return hyperparameter sensitivity.
            :param fast_mode: (bool) whether to use fast hyperparameter tuning (using holdout validation).
            :param storage_url: (str | None) filepath of the study to be used for tuning. If provided,
                                    the study will be stored on the provided directory.
            :param study_name: (str | None) name of the optuna study. Typically, the model_id is used as the name.
            :param seed: (int) random seed for sampling.
            :param open_optuna_dashboard: (bool) whether to open optuna dashboard.
        """

        # Initializing sampler algorithm.
        sampler = algorithm.create(seed=seed)

        # Initializing optuna study.
        study = optuna.create_study(
            storage=storage_url,
            sampler=sampler,
            study_name=study_name,
            direction=self._direction,
            load_if_exists=True
        )

        # Opening optuna dashboard.
        if open_optuna_dashboard and storage_url is not None:
            dashboard_process = start_optuna_dashboard(storage_url=storage_url)
            threading.Thread(target=wait_optuna_server, args=(), daemon=True).start()
        else:
            dashboard_process = None

        objective = self._objective(df=df, seed=seed) if not fast_mode else self._fast_objective(df=df, seed=seed)
        study.optimize(objective, n_trials=trials, show_progress_bar=True)

        # Extracting trial dataframe.
        trials_df = study.trials_dataframe()

        # Extracting columns of interest and renaming them appropriately.
        param_cols = [col for col in trials_df.columns if col.startswith('params_')]
        rename_dict = {
            'number': 'Trial',
            'value': self._metric,
            **{col: col.replace('params_', '') for col in param_cols}
        }
        study_df = trials_df[['number', 'value'] + param_cols]
        study_df = study_df.rename(columns=rename_dict).sort_values(by=self._metric, ascending=False, ignore_index=True)

        # Extracting feature importances.
        param_importances = optuna.importance.get_param_importances(study=study) if return_importances else None

        # Closing optuna dashboard (if open) and returning study reports.
        stop_optuna_dashboard(process=dashboard_process)
        return study_df, study.best_value, study.best_params, param_importances

    def nested_cv_tuning(
            self,
            df: pd.DataFrame,
            trials: int,
            algorithm: TuningAlgorithm,
            fast_mode: bool = False,
            seed: Optional[int] = 0
    ) -> Tuple[pd.DataFrame, float, Dict[str, float]]:
        """ Tunes the model using the nested cross validation technique. Returns the unbiased evaluation metrics of the classifier per fold,
            the unbiased evaluation score, and the parameter importances.

            Args
            ----
            :param df (pd.DataFrame) dataset to be used for tuning.
            :param trials: (int) number of trials for the hyperparameter search process.
            :param algorithm: (TuningAlgorithm) sampling algorithm to be used for tuning.
            :param fast_mode: (bool) whether to use fast hyperparameter tuning (using holdout validation).
            :param seed: (int) random seed for sampling.
        """

        nested_results = []
        nested_importances = []

        targets = construct_targets(df=df, target_type=self._model_params['target_type'])

        # Iterating outer folds.
        for i, outer_train_df, outer_test_df in cross_validation(
                df=df,
                targets=targets,
                k_folds=3,
                show_progress=False,
                seed=seed
        ):
            # Tuning model using the inner fold.
            _, best_value, best_params, importances = self.tune(
                df=outer_train_df,
                trials=trials,
                algorithm=algorithm,
                return_importances=True,
                fast_mode=fast_mode,
                storage_url=None,
                study_name=None,
                seed=seed
            )
            nested_importances.append(importances)

            # Training-Evaluating model using the outer fold.
            model_config = {**self._model_params, **best_params}
            model = self._model_cls(**model_config)
            metrics_df = model.fit(train_df=outer_train_df, eval_df=outer_test_df, seed=seed)

            eval_metrics = metrics_df.loc[metrics_df['Data'] == 'Eval']
            metric_columns = [
                column
                for column in metrics_df.columns
                if column not in {'Data', 'Start Date', 'End Date', 'Samples'}
            ]
            fold_metrics = eval_metrics.iloc[0]
            nested_results.append({
                f'Outer Fold': i+1,
                f'Best Inner {self._metric}': best_value,
                **{metric: fold_metrics[metric] for metric in metric_columns},
                **best_params
            })
        study_df = pd.DataFrame(data=nested_results)
        best_value = study_df[self._metric].mean()
        param_importances = pd.DataFrame(data=nested_importances).mean().to_dict()
        return study_df, best_value, param_importances
