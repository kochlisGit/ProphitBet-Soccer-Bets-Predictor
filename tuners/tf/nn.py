from models.model import Model
from models.tf.nn import FCNet
import pandas as pd
from tuners.tuner import Tuner


class FCNetTuner(Tuner):
    def __init__(
            self,
            n_trials: int,
            metric,
            matches_df: pd.DataFrame,
            num_eval_samples: int,
            epochs: int,
            early_stopping_epochs: int,
            learning_rate_decay_factor: float,
            learning_rate_decay_epochs: int,
            random_seed: int
    ):
        super().__init__(
            n_trials=n_trials,
            metric=metric,
            matches_df=matches_df,
            one_hot=True,
            num_eval_samples=num_eval_samples,
            random_seed=random_seed
        )

        self._epochs = epochs
        self._early_stopping_epochs = early_stopping_epochs
        self._learning_rate_decay_factor = learning_rate_decay_factor
        self._learning_rate_decay_epochs = learning_rate_decay_epochs

    def _create_model(self, trial) -> Model:
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])

        model = FCNet(
            input_shape=self.x_train.shape[1:],
            random_seed=self.random_seed
        )

        noise_range = trial.suggest_float('noise_range', low=0.0, high=0.4)

        num_hidden_layers = trial.suggest_int('num_hidden_layers', low=1, high=5, step=1)
        hidden_layers = [trial.suggest_categorical(
            f'layer_{i}', [16, 32, 64, 128, 256]) for i in range(num_hidden_layers)
        ]
        batch_normalizations = [
            trial.suggest_categorical(f'bn_{i}', [False, True]) for i in range(num_hidden_layers)
        ]
        activations = [
            trial.suggest_categorical(f'activation_{i}', [None, 'tanh', 'relu', 'gelu']) for i in range(num_hidden_layers)
        ]
        regularizations = [
            trial.suggest_categorical(f'regularization_{i}', [None, 'l1', 'l2']) for i in range(num_hidden_layers)
        ]
        dropouts = [
            trial.suggest_float(f'dropout_{i}', low=0.0, high=0.4, step=0.1) for i in range(num_hidden_layers)
        ]
        optimizer = trial.suggest_categorical(f'optimizer', ['adam', 'yogi', 'adamw'])
        learning_rate = trial.suggest_float(f'learning_rate', low=0.0001, high=0.001, step=0.0001)

        model.build_model(
            epochs=self._epochs,
            batch_size=batch_size,
            early_stopping_epochs=self._early_stopping_epochs,
            learning_rate_decay_factor=self._learning_rate_decay_factor,
            learning_rate_decay_epochs=self._learning_rate_decay_epochs,
            noise_range=noise_range,
            hidden_layers=hidden_layers,
            batch_normalizations=batch_normalizations,
            activations=activations,
            regularizations=regularizations,
            dropouts=dropouts,
            optimizer=optimizer,
            learning_rate=learning_rate
        )
        return model

    def tune(self) -> dict:
        best_params = super().tune()

        best_params['epochs'] = self._epochs
        best_params['early_stopping_epochs'] = self._early_stopping_epochs
        best_params['learning_rate_decay_rate'] = self._learning_rate_decay_factor
        best_params['learning_rate_decay_epochs'] = self._learning_rate_decay_epochs
        return best_params
