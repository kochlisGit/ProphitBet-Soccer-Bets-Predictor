import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.base import BaseEstimator, ClassifierMixin
from models.model import ScikitModel
from models.tasks import ClassificationTask


class NeuralNetwork(ScikitModel):
    class TFModel(BaseEstimator, ClassifierMixin):
        def __init__(
                self,
                tf_model: tf.keras.Model,
                batch_size: int,
                epochs: int,
                early_stopping_patience: int,
                learning_rate_patience: int,
                verbose: bool
        ):
            self.tf_model = tf_model
            self._batch_size = batch_size
            self._epochs = epochs
            self._early_stopping_patience = early_stopping_patience
            self._learning_rate_patience = learning_rate_patience
            self._verbose = verbose

            self._x_test = None
            self._y_test = None

        def set_test_data(self, x: np.ndarray, y: np.ndarray):
            self._x_test = x
            self._y_test = y

        def _get_callbacks(self):
            callbacks = []

            if self._early_stopping_patience > 0:
                callbacks.append(tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=self._early_stopping_patience,
                    restore_best_weights=True,
                    verbose=int(self._verbose)
                ))

            if self._learning_rate_patience > 0:
                callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_accuracy',
                    factor=0.2,
                    patience=self._early_stopping_patience,
                    verbose=int(self._verbose)
                ))

            return callbacks if len(callbacks) > 0 else None

        def fit(self, x: np.ndarray, y: np.ndarray):
            def one_hot_targets(targets: np.ndarray, num_classes: int) -> np.ndarray:
                return np.eye(num_classes)[targets]

            assert self._x_test is not None and self._y_test is not None, 'x_test and y_test are not provided'
            assert x.shape[1] == self._x_test.shape[1]
            assert y.ndim == 1, f'Expected Train targets to be 1D array before one-hot encoding, got {y.shape}'
            assert self._y_test.ndim == 1, \
                f'Expected Test targets to be 1D array before one-hot encoding, got {self._y_test.shape}'

            num_outputs = self.tf_model.output_shape[1]

            if num_outputs > 1:
                y = one_hot_targets(targets=y, num_classes=num_outputs)
                self._y_test = one_hot_targets(targets=self._y_test, num_classes=num_outputs)

            callbacks = self._get_callbacks()
            self.tf_model.fit(
                x,
                y,
                batch_size=self._batch_size,
                epochs=self._epochs,
                callbacks=callbacks,
                validation_data=(self._x_test, self._y_test),
                shuffle=True,
                verbose='auto' if self._verbose else 0
            )

            self._x_test = None
            self._y_test = None

        def predict(self, x: np.ndarray):
            return self.tf_model.predict(x).argmax(axis=1)

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            return self.tf_model.predict(x)

    def __init__(
            self,
            model_id: str,
            fc_hiddens: list[int] or None = None,
            activation_fn: str = 'tanh',
            weight_regularization: str or None = None,
            batch_normalization: bool = False,
            dropout_rate: float = 0.0,
            batch_size: int = 32,
            epochs: int = 100,
            early_stopping_patience: int = 50,
            learning_rate: float = 0.001,
            learning_rate_patience: int = 15,
            label_smoothing: bool = False,
            input_noise: float = 0.0,
            optimizer: str = 'adam',
            summary: bool = True,
            verbose: bool = True,
            calibrate_probabilities: bool = False,
            **kwargs
    ):
        if fc_hiddens is None:
            fc_hiddens = [256, 256]

        assert activation_fn == 'sigmoid' or activation_fn == 'relu' or activation_fn =='tanh' or activation_fn == 'gelu' or activation_fn == 'elu', \
            f'activation_fn not supported: "{activation_fn}"'
        assert weight_regularization is None or weight_regularization == 'None' or weight_regularization == 'l1' or weight_regularization == 'l2' or \
               weight_regularization == 'l1_l2', f'weight_regularization should be None, l1, l2 or l1_l2, got {weight_regularization}'
        assert 0 <= dropout_rate < 1, f'dropout rate is expected to be between 0.0 and 1.0, got {dropout_rate}'
        assert 0 <= input_noise < 1, f'Input noise is expected to be between 0.0 and 1.0, got {input_noise}'

        assert epochs > 0, f'Epochs is expected to be positive integer, got {epochs}'
        assert batch_size > 0, f'Batch size is expected to be positive integer, got {batch_size}'
        assert 0 < learning_rate < 1, f'learning rate is expected to be between 0 and 1.0, got {learning_rate}'
        assert not calibrate_probabilities, 'Probability calibration is not supported'

        self._fc_hiddens = fc_hiddens
        self._activation_fn = activation_fn
        self._weight_regularization = None if weight_regularization == 'None' else weight_regularization
        self._batch_normalization = batch_normalization
        self._dropout_rate = dropout_rate
        self._batch_size = batch_size
        self._epochs = epochs
        self._early_stopping_patience = early_stopping_patience
        self._learning_rate = learning_rate
        self._learning_rate_patience = learning_rate_patience
        self._label_smoothing = label_smoothing
        self._input_noise = input_noise
        self._optimizer = optimizer
        self._summary = summary
        self._verbose = verbose

        self._x_test = None
        self._y_test = None

        super().__init__(
            model_id=model_id,
            model_name=self._get_model_name(),
            calibrate_probabilities=False,
            **kwargs
        )

    @staticmethod
    def _get_model_name() -> str:
        return 'neural-network'

    def save(self, checkpoint_directory: str):
        assert self._model is not None, 'Model has not been initialized yet'

        checkpoint_filepath = f'{checkpoint_directory}/model.h5'
        self._model.tf_model.save(checkpoint_filepath)

    def load(self, checkpoint_directory: str):
        checkpoint_filepath = f'{checkpoint_directory}/model.h5'
        tf_model = tf.keras.models.load_model(checkpoint_filepath, compile=False)

        if self._summary:
            tf_model.summary(expand_nested=True)

        self._model = NeuralNetwork.TFModel(
            tf_model=tf_model,
            batch_size=self._batch_size,
            epochs=self._epochs,
            early_stopping_patience=self._early_stopping_patience,
            learning_rate_patience=self._learning_rate_patience,
            verbose=self._verbose
        )

    def _build_neural_network(self, input_size: int, num_classes: int) -> tf.keras.Model:
        layers = [tf.keras.layers.Input(shape=(input_size,), name='inputs')]

        if self._input_noise > 0.0:
            layers.append(tf.keras.layers.GaussianNoise(stddev=self._input_noise, name='input_noise'))

        for i, units in enumerate(self._fc_hiddens):
            weight_regularization = self._weight_regularization if i == 0 else None

            use_bias = not self._batch_normalization
            layers.append(tf.keras.layers.Dense(
                units=units,
                activation=self._activation_fn,
                kernel_regularizer=weight_regularization,
                use_bias=use_bias,
                name=f'hidden_layer_{i + 1}'
            ))

            if self._batch_normalization:
                layers.append(tf.keras.layers.BatchNormalization(name=f'batch_norm_{i + 1}'))

            if self._dropout_rate > 0.0:
                layers.append(tf.keras.layers.Dropout(rate=self._dropout_rate, name=f'dropout_{i + 1}'))

        if num_classes == 2:
            num_outputs = 1
            final_activation = 'sigmoid'
        else:
            num_outputs = num_classes
            final_activation = 'softmax'

        layers.append(tf.keras.layers.Dense(units=num_outputs, activation=final_activation, name='outputs'))
        return tf.keras.Sequential(layers)

    def _get_loss(self, num_classes: int):
        if num_classes == 2:
            loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1 if self._label_smoothing else 0.0)
        else:
            assert num_classes > 2, f'num_classes should be > 2 to support categorical_crossentropy, got {num_classes}'

            loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1 if self._label_smoothing else 0.0)

        return loss

    def _compile_neural_network(self, model: tf.keras.Model, num_classes: int) -> tf.keras.Model:
        if self._optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
        elif self._optimizer == 'radam':
            optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self._learning_rate, weight_decay=0.001)
        elif self._optimizer == 'adabelief':
            optimizer = tfa.optimizers.AdaBelief(learning_rate=self._learning_rate, weight_decay=0.001)
        elif self._optimizer == 'lookahead-adabelief':
            optim = tfa.optimizers.AdaBelief(learning_rate=self._learning_rate, weight_decay=0.001, amsgrad=True)
            optimizer = tfa.optimizers.Lookahead(optimizer=optim, sync_period=10)
        else:
            raise NotImplementedError(f'Not implemented optimizer: {self._optimizer}')

        loss = self._get_loss(num_classes=num_classes)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        return model

    def _build_estimator(self, input_size: int, num_classes: int) -> BaseEstimator:
        tf_model = self._build_neural_network(input_size=input_size, num_classes=num_classes)
        tf_model = self._compile_neural_network(model=tf_model, num_classes=num_classes)

        if self._summary:
            tf_model.summary(expand_nested=True)

        tf_model = NeuralNetwork.TFModel(
            tf_model=tf_model,
            batch_size=self._batch_size,
            epochs=self._epochs,
            early_stopping_patience=self._early_stopping_patience,
            learning_rate_patience=self._learning_rate_patience,
            verbose=self._verbose
        )
        tf_model.set_test_data(x=self._x_test, y=self._y_test)
        return tf_model

    def fit(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
            task: ClassificationTask,
            add_classification_report: bool
    ) -> (dict[str, float], str):
        self._x_test = x_test
        self._y_test = y_test
        return super().fit(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            task=task,
            add_classification_report=add_classification_report
        )
