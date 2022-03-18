import os
import numpy as np
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.activations as activations
import tensorflow_addons.optimizers as optimizers


class FCModel:
    def __init__(self, model_name):
        self._default_checkpoint_dir = 'models/checkpoints/{}'.format(model_name)
        self._num_targets = 3
        self._default_batch_size = 16

        self._model = None

        self._initialize_model_directory()

    def _initialize_model_directory(self):
        if not os.path.exists(self._default_checkpoint_dir):
            os.makedirs(self._default_checkpoint_dir)

    def build_model(self, input_shape, hidden_layers):
        model = models.Sequential()
        model.add(layers.Input(input_shape))

        model.add(layers.Dense(units=hidden_layers[0], activation=None, use_bias=False, kernel_regularizer='l2'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(activations.gelu))

        for units in hidden_layers[1:]:
            model.add(layers.Dense(units=units, activation=None, use_bias=False))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation(activations.gelu))
            model.add(layers.Dropout(rate=0.4))

        model.add(layers.Dense(units=self._num_targets, activation='softmax', use_bias=True, kernel_regularizer='l1'))
        model.compile(
            optimizer=optimizers.Yogi(learning_rate=0.0001),
            loss='huber',
            metrics=['accuracy']
        )
        self._model = model

    def save(self):
        self._model.save(self._default_checkpoint_dir)

    def load(self):
        self._model = models.load_model(self._default_checkpoint_dir)

    def train(self, x_train, y_train, x_test, y_test, odd_noise_range):

        odd_noise = np.random.uniform(
            low=-odd_noise_range,
            high=odd_noise_range,
            size=(x_train.shape[0], self._num_targets)
        )
        x = x_train.copy()
        x[:, 0:3] += odd_noise

        history = self._model.fit(
            x,
            y_train,
            batch_size=self._default_batch_size,
            validation_data=(x_test, y_test),
            epochs=1,
            verbose=0
        )

        accuracy = history.history['val_accuracy'][-1]
        return accuracy

    def predict(self, x):
        return self._model.predict(x)
