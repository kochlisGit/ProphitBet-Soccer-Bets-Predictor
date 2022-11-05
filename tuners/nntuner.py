from models.neuralnet.nn import FCNet
from sklearn.metrics import accuracy_score
import numpy as np
import optuna
import tensorflow as tf


class NNTuner:
    def __init__(
            self,
            checkpoint_path: str,
            league_identifier: str,
            model_name: str,
            epochs: int,
            early_stopping_patience: int,
            n_trials: int,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray
    ):
        self._checkpoint_path = checkpoint_path
        self._league_identifier = league_identifier
        self._model_name = f'{model_name}'
        self._n_trials = n_trials
        self._epochs = epochs
        self._early_stopping_patience = early_stopping_patience
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test

    def _objective(self, trial) -> float:
        hidden_layers = []

        n_layers = trial.suggest_int('n_layers', 1, 4)
        for i in range(n_layers):
            units = trial.suggest_int(f'hidden_units_{i}', 16, 256)
            hidden_layers.append(units)

        noise_1 = trial.suggest_uniform('noise_1', 0.0, 0.3)
        noise_x = trial.suggest_uniform('noise_x', 0.0, 0.3)
        noise_2 = trial.suggest_uniform('noise_2', 0.0, 0.3)

        noise_favorites_only = bool(trial.suggest_int('noise_favorites_only', 0, 1))
        win_draw_noise = bool(trial.suggest_int('win_draw_noise', 0, 1))

        batch_normalization = bool(trial.suggest_int('batch_normalization', 0, 1))
        dropout = trial.suggest_categorical('dropout', [None, 0.1, 0.2, 0.3, 0.4])
        regularization = trial.suggest_categorical('regularization', [None, 'l1', 'l2'])
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'yogi', 'adamw'])
        learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.001, 0.0005, 0.0002])

        tf.random.set_seed(0)
        model = FCNet(
            input_shape=self._x_train.shape[1:],
            checkpoint_path=self._checkpoint_path,
            league_identifier=self._league_identifier,
            model_name=self._model_name
        )
        model.build_model(
            hidden_layers=hidden_layers,
            batch_normalization=batch_normalization,
            dropout=dropout,
            regularization=regularization,
            optimizer=optimizer,
            learning_rate=learning_rate
        )

        model.train(
            x_train=self._x_train,
            y_train=self._y_train,
            x_test=self._x_test,
            y_test=self._y_test,
            odd_noise_ranges=np.float32([noise_1, noise_x, noise_2]),
            performance_rate_noise=win_draw_noise,
            noise_favorites_only=noise_favorites_only,
            epochs=self._epochs,
            early_stopping_epochs=self._early_stopping_patience
        )

        _, y_pred = model.predict(x_inputs=self._x_test)
        return accuracy_score(np.argmax(self._y_test, axis=1), y_pred)

    def tune(self) -> dict:
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=self._n_trials)

        best_params = study.best_trial.params
        best_params['epochs'] = self._epochs
        best_params['early_stopping_patience'] = self._early_stopping_patience
        return best_params
