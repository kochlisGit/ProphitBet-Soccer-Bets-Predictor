import threading
import optuna
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from src.core.models.dashboard import start_optuna_dashboard, stop_optuna_dashboard, wait_optuna_server
from src.core.models.tuner import TuningAlgorithm
from src.core.preprocessing.utils.selection import train_test_split


class Tuner:
    """ Classification Model tuner class.

        Attributes
        ------
        :param agent_cls: (Type[Classifier]) classifier's instance to be created.
        :param agent_params: fixed model parameters (remain frozen during the tuning process).
        :param tunable_params: tunable model parameters (the hyperparameters of the model that will be tuned).
    """

    def __init__(
            self,
            agent_cls: Type,
            agent_params: Dict[str, Any],
            tunable_params: Dict[str, Union[List, Dict[str, Any]]]
    ):
        if len(tunable_params) == 0:
            raise ValueError('tunable_params does not contain any tunable parameters. Tuning is aborted.')

        self._agent_cls = agent_cls
        self._agent_params = agent_params
        self._tunable_params = tunable_params

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

    def _objective(self, df: pd.DataFrame, test_size: Union[float, int], seed: Optional[int]) -> Callable:
        """ Optuna's objective function. It uses holdout validation to optimize the agent's policy.

            Args
            ----
            :param df: (pd.DataFrame) dataset to be used for optimization.
            :param seed: (bool) cross-validation random seed.
        """

        def objective(trial: optuna.Trial) -> float:
            trial_params = self._suggest_params(trial=trial)
            agent_config = {**self._agent_params, **trial_params}
            agent = self._agent_cls(**agent_config)
            train_df, eval_df = train_test_split(df=df, test_size=test_size)
            metrics_df = agent.train(train_df=train_df, eval_df=eval_df, seed=seed, verbose=0, trial=trial)
            rewards = metrics_df['Avg. Eval Rewards'].max()
            return rewards
        return objective

    def tune(
            self,
            df: pd.DataFrame,
            test_size: Union[float, int],
            trials: int,
            algorithm: TuningAlgorithm,
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
            :param test_size: (float | int) test size of match data to be used for evaluation.
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
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=self._agent_params['eval_freq'],
                max_resource=self._agent_params['train_steps'],
            ),
            study_name=study_name,
            direction='maximize',
            load_if_exists=True
        )

        # Opening optuna dashboard.
        if open_optuna_dashboard and storage_url is not None:
            dashboard_process = start_optuna_dashboard(storage_url=storage_url)
            threading.Thread(target=wait_optuna_server, args=(), daemon=True).start()
        else:
            dashboard_process = None

        study.optimize(self._objective(df=df, test_size=test_size, seed=seed), n_trials=trials, show_progress_bar=True)

        # Extracting trial dataframe.
        trials_df = study.trials_dataframe()

        # Extracting columns of interest and renaming them appropriately.
        param_cols = [col for col in trials_df.columns if col.startswith('params_')]
        rename_dict = {
            'number': 'Trial',
            'value': 'Rewards',
            **{col: col.replace('params_', '') for col in param_cols}
        }
        study_df = trials_df[['number', 'value'] + param_cols]
        study_df = study_df.rename(columns=rename_dict).sort_values(by='Rewards', ascending=False, ignore_index=True)

        # Extracting feature importances.
        param_importances = optuna.importance.get_param_importances(study=study)

        # Closing optuna dashboard (if open) and returning the study reports.
        stop_optuna_dashboard(process=dashboard_process)
        return study_df, study.best_value, study.best_params, param_importances
