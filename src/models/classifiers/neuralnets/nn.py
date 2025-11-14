import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from sklearn.base import TransformerMixin
from src.preprocessing.utils.sampling import SamplerType
from src.preprocessing.utils.target import TargetType
from src.models.classifiers.neuralnets.tfmodel import TFModel
from src.models.model import ClassificationModel


class NeuralNetwork(ClassificationModel):
    """ Implementation of Neural Network Classifier. It does not support probability calibration.

        Parameters
        -----------------------------

        :param league_id: The created league id.
        :param model_id: The provided model id by the user.
        :param target_type: The classification target type.
        :param normalizer: The input data normalizer placeholder.
        :param sampler: The sampler method for training data.
        :param hidden_layers: int. Number of hidden layers (Neuron layers).
        :param hidden_units: int. Number of units (Neurons) per layer.
        :param vsn: bool. Whether to use a Variable Selection Network block to highlight important features.
        :param layer_normalization: bool. Whether to apply layer normalization after VSN
        :param batch_normalization: bool. Whether to apply batch normalization after each hidden layer.
        :param dropout_rate: float. Whether to apply dropout regularization with the specified rate (0.0 to 1.0).
        :param odd_noise_std: float. Whether to apply noise regularization to odd inputs with the specified rate.
        :param class_weight: bool. Whether to balance class weights.
        :param optimizer: str. Training optimization method. Supports ('adam', 'adabelief', 'adan', 'ranger25').
        :param lookahead: bool. Whether to use the lookahead optimization technique.
        :param label_smoothing: float. Whether to apply label smoothing with the specified smoothing factor.
        :param learning_rate: float. Optimizer's initial learning rate.
        :param batch_size: int. Optimizer's batch size (number of inputs per update step).
        :param early_stopping_patience: int. Number of epochs which loss does not improve before early-stopping kicks in.
        :param lr_decay_patience: int. Number of epochs which loss does not improve before learning rate decays.
        :param lr_decay_factor: int. Learning rate decay factor once decay mechanism kicks in.
        :param verbose: str | int. The verbose mode during training (whether to log the training progress bar to cmd).
        :param input_size: Number of input features.
        :param num_classes: Number of output classes.
    """

    def __init__(
            self,
            league_id: str,
            model_id: str,
            target_type: TargetType,
            normalizer: Optional[TransformerMixin] = None,
            sampler: Optional[SamplerType] = None,
            eval_odds_filter: Optional[Dict] = None,
            num_eval_samples: Optional[int] = None,
            hidden_layers: int = 1,
            hidden_units: int = 256,
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
            verbose: Union[str, int] = 'auto',
            input_size: Optional[int] = None,
            num_classes: Optional[int] = None,
            **kwargs
    ):
        self._hidden_layers = hidden_layers
        self._hidden_units = hidden_units
        self._hidden_activation = hidden_activation
        self._vsn = vsn
        self._layer_normalization = layer_normalization
        self._batch_normalization = batch_normalization
        self._dropout_rate = dropout_rate
        self._odd_noise_std = odd_noise_std
        self._class_weight = class_weight
        self._optimizer = optimizer
        self._lookahead = lookahead
        self._label_smoothing = label_smoothing
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._epochs = epochs
        self._early_stopping_patience = early_stopping_patience
        self._lr_decay_patience = lr_decay_patience
        self._lr_decay_factor = lr_decay_factor
        self._verbose = verbose

        self._input_size = input_size
        self._num_classes = num_classes

        super().__init__(
            league_id=league_id,
            model_id=model_id,
            target_type=target_type,
            calibrate_probabilities=False,
            normalizer=normalizer,
            sampler=sampler,
            eval_odds_filter=eval_odds_filter,
            num_eval_samples=num_eval_samples,
            **kwargs
        )

    @property
    def input_size(self) -> Optional[int]:
        return self._input_size

    @property
    def num_classes(self) -> Optional[int]:
        return self._num_classes

    def load(self, checkpoint_directory: str):
        """ Loads the classifier using pickle. """

        if self._input_size is None:
            return

        self._classifier = self.build_classifier(input_size=self._input_size, num_classes=self._num_classes)

        if os.path.exists(path=f'{checkpoint_directory}/model.ckpt'):
            self._classifier.model.load_weights(filepath=f'{checkpoint_directory}/model.ckpt')
        if os.path.exists(path=f'{checkpoint_directory}/attn_model.ckpt'):
            self._classifier.model.load_weights(filepath=f'{checkpoint_directory}/attn_model.ckpt')

    def save(self, checkpoint_directory: str):
        """ Saves the classifier using pickle. """

        if self._classifier is None:
            raise ValueError('Classifier has not been built/fit yet. Cannot save None classifier.')

        self._classifier.model.save_weights(filepath=f'{checkpoint_directory}/model.ckpt')

        if self._classifier.attn_model is not None:
            self._classifier.attn_model.save_weights(filepath=f'{checkpoint_directory}/attn_model.ckpt')

    def fit(self, train_df: pd.DataFrame, eval_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """ Fits the classifier in the data and returns the evaluation report.
            :param train_df: The train dataframe used for fitting.
            :param eval_df: The eval dataframe used for evaluation.
        """

        # Preprocess both train and eval set.
        x_train, y_train, self._normalizer = self._dataset_preprocessor.preprocess_dataset(
            df=train_df,
            target_type=self._target_type,
            normalizer=self._normalizer,
            sampler=self._sampler,
            seed=0
        )
        x_eval, y_eval, _ = self._dataset_preprocessor.preprocess_dataset(
            df=eval_df if eval_df is not None else train_df,
            target_type=self._target_type,
            normalizer=self._normalizer
        )

        # Build & Fit classifier.
        self._classifier = self.build_classifier(input_size=x_train.shape[1], num_classes=np.unique(y_train).shape[0])
        self._classifier.fit(x_train, y_train, x_eval=x_eval, y_eval=y_eval)

        # Evaluate classifier.
        if eval_df is not None:
            self._num_eval_samples = eval_df.shape[0]

        metrics_df = self._evaluate_classifier(train_df=train_df, eval_df=eval_df)
        return metrics_df

    def build_classifier(self, input_size: int, num_classes: int) -> TFModel:
        """ Builds a Deep Neural Network for classification tasks. """

        if self._input_size is None:
            self._input_size = input_size
        if self._num_classes is None:
            self._num_classes = num_classes

        return TFModel(
            num_inputs=input_size,
            num_classes=num_classes,
            target_type=self._target_type,
            hidden_layers=self._hidden_layers,
            hidden_units=self._hidden_units,
            hidden_activation=self._hidden_activation,
            vsn=self._vsn,
            layer_normalization=self._layer_normalization,
            batch_normalization=self._batch_normalization,
            dropout_rate=self._dropout_rate,
            odd_noise_std=self._odd_noise_std,
            class_weight=self._class_weight,
            optimizer=self._optimizer,
            label_smoothing=self._label_smoothing,
            learning_rate=self._learning_rate,
            batch_size=self._batch_size,
            epochs=self._epochs,
            early_stopping_patience=self._early_stopping_patience,
            lr_decay_patience=self._lr_decay_patience,
            lr_decay_factor=self._lr_decay_factor,
            verbose=self._verbose
        )

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == 'hidden_layers':
            return {'low': 1, 'high': 4, 'step': 1}
        elif param == 'hidden_units':
            return {'low': 64, 'high': 512, 'step': 64}
        elif param == 'hidden_activation':
            return ['tanh', 'relu', 'gelu']
        elif param == 'vsn':
            return [True, False]
        elif param == 'layer_normalization':
            return [True, False]
        elif param == 'batch_normalization':
            return [True, False]
        elif param == 'dropout_rate':
            return {'low': 0.0, 'high': 0.5, 'step': 0.1}
        elif param == 'odd_noise_std':
            return {'low': 0.00, 'high': 0.2, 'step': 0.01}
        elif param == 'class_weight':
            return [True, False]
        elif param == 'optimizer':
            return ['adam', 'adabelief', 'adan', 'ranger25']
        elif param == 'lookahead':
            return [True, False]
        elif param == 'label_smoothing':
            return {'low': 0.0, 'high': 0.1, 'step': 0.01}
        elif param == 'learning_rate':
            return {'low': 0.0005, 'high': 0.02, 'step': 0.0005}
        elif param == 'batch_size':
            return {'low': 16, 'high': 128, 'step': 16}
        elif param == 'epochs':
            return {'low': 10, 'high': 100, 'step': 10}
        elif param == 'early_stopping_patience':
            return {'low': 10, 'high': 30, 'step': 5}
        elif param == 'lr_decay_patience':
            return {'low': 0, 'high': 30, 'step': 5}
        elif param == 'lr_decay_factor':
            return {'low': 0.2, 'high': 0.5, 'step': 0.1}
        else:
            raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.pop('calibrate_probabilities')
        model_config.update({
            'hidden_layers': self._hidden_layers,
            'hidden_units': self._hidden_units,
            'hidden_activation': self._hidden_activation,
            'vsn': self._vsn,
            'layer_normalization': self._layer_normalization,
            'batch_normalization': self._batch_normalization,
            'dropout_rate': self._dropout_rate,
            'odd_noise_std': self._odd_noise_std,
            'class_weight': self._class_weight,
            'optimizer': self._optimizer,
            'lookahead': self._lookahead,
            'label_smoothing': self._label_smoothing,
            'learning_rate': self._learning_rate,
            'batch_size': self._batch_size,
            'epochs': self._epochs,
            'early_stopping_patience': self._early_stopping_patience,
            'lr_decay_patience': self._lr_decay_patience,
            'lr_decay_factor': self._lr_decay_factor,
            'verbose': self._verbose,
            'input_size': self._input_size,
            'num_classes': self._num_classes
        })
        return model_config
