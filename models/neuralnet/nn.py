from models.model import Model
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


class FCNet(Model):
    def __init__(
            self,
            input_shape: tuple,
            checkpoint_path: str,
            league_identifier: str,
            model_name: str
    ):
        model_name += '_nn'
        super().__init__(
            input_shape=input_shape,
            checkpoint_path=checkpoint_path,
            league_identifier=league_identifier,
            model_name=model_name
        )

        self._max_favorite_odd_noise = 2.0

    @property
    def num_classes(self) -> int:
        return 3

    @property
    def batch_size(self) -> int:
        return 16

    def build_model(self, **kwargs):
        hidden_layers = kwargs['hidden_layers']

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(self.input_shape))

        model.add(tf.keras.layers.Dense(
            units=hidden_layers[0], activation=None, use_bias=False, kernel_regularizer='l2')
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(tf.keras.activations.gelu))

        for units in hidden_layers[1:]:
            model.add(tf.keras.layers.Dense(units=units, activation=None, use_bias=False))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Activation(tf.keras.activations.gelu))
            model.add(tf.keras.layers.Dropout(rate=0.4))

        model.add(tf.keras.layers.Dense(
            units=self.num_classes, activation='softmax', use_bias=True, kernel_regularizer='l1')
        )
        model.compile(
            optimizer=tfa.optimizers.Yogi(learning_rate=0.0005),
            loss='huber',
            metrics=['accuracy']
        )
        self._model = model

    def _save(self):
        self._model.save(self.checkpoint_directory)

    def load(self):
        self._model = tf.keras.models.load_model(self.checkpoint_directory)

    def train(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
            **kwargs,
    ) -> float:
        odd_noise_ranges = kwargs['odd_noise_ranges']
        performance_rate_noise = kwargs['performance_rate_noise']
        noise_favorites_only = kwargs['noise_favorites_only']

        x = x_train.copy()
        if noise_favorites_only:
            favorite_home_indices = x[:, 0] < self._max_favorite_odd_noise
            favorite_away_indices = x[:, 2] < self._max_favorite_odd_noise

            odd_noise = np.zeros(shape=(x_train.shape[0], self.num_classes))
            odd_noise[favorite_home_indices, 0] += np.random.uniform(
                low=-odd_noise_ranges[0],
                high=odd_noise_ranges[0],
                size=(np.sum(favorite_home_indices),)
            )
            odd_noise[favorite_away_indices, 2] += np.random.uniform(
                low=-odd_noise_ranges[2],
                high=odd_noise_ranges[2],
                size=(np.sum(favorite_away_indices,))
            )
        else:
            odd_noise = np.random.uniform(
                low=-odd_noise_ranges,
                high=odd_noise_ranges,
                size=(x_train.shape[0], self.num_classes)
            )

        x[:, 0:3] += odd_noise

        if performance_rate_noise:
            x[:, 8: 10] += np.random.uniform(
                low=-0.05,
                high=0.05,
                size=(x_train.shape[0], 2)
            )
            x[:, 15: 17] += np.random.uniform(
                low=-0.05,
                high=0.05,
                size=(x_train.shape[0], 2)
            )

        history = self._model.fit(
            x,
            y_train,
            batch_size=self.batch_size,
            validation_data=(x_test, y_test),
            epochs=1,
            verbose=0
        )

        accuracy = history.history['val_accuracy'][-1]
        return accuracy

    def predict(self, x_inputs: np.ndarray):
        predict_proba = np.round(self.model.predict(x_inputs), 2)
        y_pred = np.argmax(predict_proba, axis=1)
        return predict_proba, y_pred
