import numpy as np
import optuna
import pandas as pd
from abc import ABC, abstractmethod
from preprocessing.training import preprocess_training_dataframe, split_train_targets
from models.model import Model


class Tuner(ABC):
    def __init__(
        self,
        n_trials: int,
        metric,
        matches_df: pd.DataFrame,
        one_hot: bool,
        num_eval_samples: int,
        random_seed: int = 0,
    ):
        self._n_trials = n_trials
        self._metric = metric
        self._random_seed = random_seed
        self._one_hot = one_hot

        inputs, targets = preprocess_training_dataframe(
            matches_df=matches_df, one_hot=one_hot
        )
        self._x_train, self._y_train, self._x_test, self._y_test = split_train_targets(
            inputs=inputs, targets=targets, num_eval_samples=num_eval_samples
        )

    @property
    def random_seed(self) -> int:
        return self._random_seed

    @property
    def x_train(self) -> np.ndarray:
        return self._x_train

    @property
    def y_train(self) -> np.ndarray:
        return self._y_train

    @property
    def x_test(self) -> np.ndarray:
        return self._x_test

    @property
    def y_test(self) -> np.ndarray:
        return self._y_test

    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y = np.argmax(y_true, axis=1) if self._one_hot else y_true
        return self._metric(y_true=y, y_pred=y_pred)

    @abstractmethod
    def _create_model(self, trial) -> Model:
        pass

    def _objective(self, trial) -> float:
        model = self._create_model(trial=trial)
        use_over_sampling = bool(
            trial.suggest_categorical("user_over_sampling", [True, False])
        )

        model.train(
            x_train=self.x_train,
            y_train=self.y_train,
            x_test=self.x_test,
            y_test=self.y_test,
            use_over_sampling=use_over_sampling,
        )
        y_pred, _ = model.predict(x=self.x_test)
        return self._evaluate(y_true=self.y_test, y_pred=y_pred)

    def tune(self) -> dict:
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self._n_trials)
        best_params = study.best_trial.params
        return best_params
