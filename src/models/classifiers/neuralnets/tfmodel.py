import random
import numpy as np
import tensorflow as tf
from typing import List, Optional, Tuple, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from src.preprocessing.utils.target import TargetType, one_hot_encode
from src.models.classifiers.neuralnets import optimizers
from src.models.classifiers.neuralnets.layers.vsn import VariableSelectionNetwork


class TFModel(ClassifierMixin, BaseEstimator):
    """ Implementation of scikit-learn-based Neural Network.

        Parameters
        --------------------------------------

        :param num_inputs: int. Number of dataset features (inputs).
        :param num_classes: int. Number of dataset classes. If num_classes == 2, then the binary cross entropy is used.
        :param target_type: TargetType. The classification task target type.
        :param hidden_layers: int. Number of hidden layers (Neuron layers).
        :param hidden_units: int. Number of units (Neurons) per layer.
        :param vsn: bool. Whether to use a Variable Selection Network block to highlight important features.
        :param layer_normalization: bool. Whether to apply layer normalization after VSN
        :param batch_normalization: bool. Whether to apply batch normalization after each hidden layer.
        :param dropout_rate: float. Whether to apply dropout regularization with the specified rate (0.0 to 1.0).
        :param odd_noise_std: float. Whether to apply noise regularization to odd inputs with the specified rate.
        :param class_weight: bool. Whether to balance class weights.
        :param optimizer: str. Training optimization method. Supports ('adam', 'adabelief', 'adan', 'ranger25').
        :param lookahead: bool. Whether to use lookahead optimization technique.
        :param label_smoothing: float. Whether to apply label smoothing with the specified smoothing factor.
        :param learning_rate: float. Optimizer's initial learning rate.
        :param batch_size: int. Optimizer's batch size (number of inputs per update step).
        :param early_stopping_patience: int. Number of epochs which loss does not improve before early-stopping kicks in.
        :param lr_decay_patience: int. Number of epochs which loss does not improve before learning rate decays.
        :param lr_decay_factor: int. Learning rate decay factor once decay mechanism kicks in.
        :param verbose: str | int. The verbose mode during training (whether to log the training progress bar to cmd).
     """

    def __init__(
            self,
            num_inputs: int,
            num_classes: int,
            target_type: TargetType,
            hidden_layers: int = 1,
            hidden_units: int = 128,
            hidden_activation: str = 'gelu',
            vsn: bool = True,
            layer_normalization: bool = True,
            batch_normalization: bool = True,
            dropout_rate: float = 0.2,
            odd_noise_std: float = 0.1,
            class_weight: bool = True,
            optimizer: str = 'adam',
            lookahead: bool = True,
            label_smoothing: float = 0.1,
            learning_rate: float = 0.001,
            batch_size: int = 16,
            epochs: int = 100,
            early_stopping_patience: int = 30,
            lr_decay_patience: int = 20,
            lr_decay_factor: float = 0.2,
            verbose: Union[str, int] = 'auto'
    ):
        BaseEstimator.__init__(self)
        ClassifierMixin.__init__(self)

        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.target_type = target_type
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation
        self.vsn = vsn
        self.layer_normalization = layer_normalization
        self.batch_normalization = batch_normalization
        self.dropout_rate = dropout_rate
        self.odd_noise_std = odd_noise_std
        self.class_weight = class_weight
        self.optimizer = optimizer
        self.lookahead = lookahead
        self.label_smoothing = label_smoothing
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.lr_decay_patience = lr_decay_patience
        self.lr_decay_factor = lr_decay_factor
        self.verbose = verbose

        # Check whether classification task is binary classification or multiclass.
        if num_classes == 2:
            self._num_outputs = 1
            self._output_activation = 'sigmoid'
            self._loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
        else:
            self._num_outputs = num_classes
            self._output_activation = 'softmax'
            self._loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

        self.model, self.attn_model = self._build_model()
        self.history = None

        # Classes placeholder.
        self.classes_ = np.arange(0, stop=num_classes, step=1, dtype=np.int32)

    def _build_model(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """ Builds the neural network model. If VSN, it also outputs the attention importances model. """

        # Set random seeds.
        tf.random.set_seed(seed=0)
        np.random.seed(seed=0)
        random.seed(0)

        inputs = tf.keras.layers.Input(shape=(self.num_inputs,), name='inputs')

        if self.odd_noise_std > 0.0:
            odds, feats = tf.split(value=inputs, num_or_size_splits=[3, self.num_inputs - 3], axis=-1)
            noisy_odds = tf.keras.layers.GaussianNoise(stddev=self.odd_noise_std, name='odd_noise')(odds)
            x = tf.keras.layers.Concatenate(axis=-1, name='input_concat')([noisy_odds, feats])
        else:
            x = inputs

        if self.vsn:
            x, attn_weights = VariableSelectionNetwork(
                num_inputs=self.num_inputs,
                hidden_units=self.hidden_units,
                dropout_rate=self.dropout_rate,
                name='vsn'
            )(x)
        else:
            attn_weights = None

        if self.layer_normalization:
            x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)

        for i in range(self.hidden_layers):
            x = tf.keras.layers.Dense(
                units=self.hidden_units,
                activation=self.hidden_activation,
                name=f'hidden_linear_{i+1}'
            )(x)

            if self.batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            if self.dropout_rate:
                x = tf.keras.layers.Dropout(rate=self.dropout_rate, name=f'hidden_dropout_{i+1}')(x)

        y = tf.keras.layers.Dense(units=self._num_outputs, activation=self._output_activation, name='outputs')(x)

        model = tf.keras.Model(inputs=inputs, outputs=y, name='NeuralNetwork')

        if self.verbose != 0:
            model.summary(expand_nested=True)

        attn_model = tf.keras.Model(inputs=inputs, outputs=attn_weights, name='AttentionModel') if attn_weights is not None else None
        return model, attn_model

    def fit(self, x: np.ndarray, y: np.ndarray, x_eval: Optional[np.ndarray] = None, y_eval: Optional[np.ndarray] = None):
        """ Implementation of scikit-learn x, y. Optionally, it uses x_eval and y_eval inputs for runtime validation. """

        # Gets class weights (optionally), which is a Dict of [label, weight].
        if self.class_weight:
            weights = compute_class_weight(class_weight='balanced', classes=self.classes_, y=y)
            class_weights = dict(enumerate(weights))
        else:
            class_weights = None

        # One-hot encoding targets if classification task is multiclass.
        if self._num_outputs > 1:
            y_encoded = one_hot_encode(y=y, target_type=self.target_type)
            y_eval_encoded = one_hot_encode(y=y_eval, target_type=self.target_type) if y_eval is not None else y_eval
        else:
            y_encoded = y
            y_eval_encoded = y_eval

        self._compile()
        self.history = self.model.fit(
            x=x,
            y=y_encoded,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=self._get_callbacks(),
            validation_data=(x, y_encoded) if x_eval is None else (x_eval, y_eval_encoded),
            shuffle=True,
            class_weight=class_weights
        )

    def predict_proba(self,  x: np.ndarray) -> np.ndarray:
        """ Implementation of scikit-learn predict_proba. It outputs the class probabilities for each class. """

        probabilities = self.model.predict(x, batch_size=self.batch_size)

        if self._num_outputs == 1:
            negative_probs = 1.0 - probabilities
            return np.hstack((negative_probs, probabilities))

        return probabilities

    def predict(self, x: np.ndarray) -> np.ndarray:
        """ Implementation of scikit-learn predict. It outputs the predicted class. """

        probabilities = self.model.predict(x, batch_size=self.batch_size)

        if self._num_outputs == 1:
            return np.squeeze((probabilities >= 0.5).astype(int), axis=-1)

        return probabilities.argmax(axis=1)

    def get_feature_importances(self, x: np.ndarray) -> Optional[np.ndarray]:
        """ Returns the feature importance if VSN is used. The importances have shape [B x F] """

        if self.attn_model is None:
            return None

        return self.attn_model(x)

    def _compile(self):
        """ Compiles the model before training. """

        optimizer = self._get_optimizer()
        self.model.compile(optimizer=optimizer, loss=self._loss, metrics=['accuracy'])

    def _get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """ Builds the optimizer. """

        if self.optimizer == 'adabelief':
            base_optimizer = optimizers.AdaBelief(learning_rate=self.learning_rate)
        elif self.optimizer == 'adam':
            base_optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'adan':
            base_optimizer = optimizers.Adan(learning_rate=self.learning_rate)
        elif self.optimizer == 'ranger25':
            base_optimizer = optimizers.Ranger25(learning_rate=self.learning_rate)
        else:
            raise ValueError(f'Undefined optimizer: "{self.optimizer}".')

        optimizer = optimizers.Lookahead(optimizer=base_optimizer) if self.lookahead else base_optimizer
        return optimizer

    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """ Gets model callbacks during training. """

        callbacks = [tf.keras.callbacks.TerminateOnNaN()]

        if self.early_stopping_patience > 0:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.early_stopping_patience))
        if self.lr_decay_patience > 0 and self.lr_decay_factor > 0.0:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=self.lr_decay_factor))

        return callbacks
