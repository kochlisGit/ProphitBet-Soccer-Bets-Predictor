import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from sklearn.base import TransformerMixin
from sklearn.utils.class_weight import compute_class_weight
from src.core.preprocessing.utils import NormalizerType, SamplerType, TargetType
from src.core.models.classifiers.neuralnets.tfmodel import TFModel
from src.core.models.model import Classifier


class NeuralNetwork(Classifier):
    """ Implementation of Neural Network Classifier. It does not support probability calibration.

        Algorithms:
            1. Gaussian assumes that each feature follows a Normal distribution.
            2. Multinomial assumes  that each feature follows a Multinomial distribution.
            3. Complement is good for imbalanced classes.

       Attributes
       -----------
        :param target_type: The classification target type.
        :param normalizer: (str | NormalizerType | TransformMixIn) input data normalizer placeholder. If NormalizerType provided,
                           then the normalizer is built and stored once fit method is called.
        :param sampler: (str | SamplerType | BaseSampler) sampler method for training data.
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
            target_type: TargetType,
            normalizer: Optional[Union[NormalizerType, TransformerMixin]] = None,
            sampler: Optional[SamplerType] = None,
            input_size: Optional[int] = None,
            num_classes: Optional[int] = None,
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
            epochs: int = 30,
            early_stopping_patience: int = 20,
            lr_decay_patience: int = 20,
            lr_decay_factor: float = 0.2,
            verbose: Union[str, int] = 'auto',
            **kwargs
    ):
        super().__init__(
            target_type=target_type,
            calibrate_probabilities=False,
            n_jobs=-1,
            normalizer=normalizer,
            sampler=sampler,
            input_size=input_size,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            vsn=vsn,
            layer_normalization=layer_normalization,
            batch_normalization=batch_normalization,
            dropout_rate=dropout_rate,
            odd_noise_std=odd_noise_std,
            class_weight=class_weight,
            optimizer=optimizer,
            lookahead=lookahead,
            label_smoothing=label_smoothing,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            lr_decay_patience=lr_decay_patience,
            lr_decay_factor=lr_decay_factor,
            verbose=verbose
        )

    @staticmethod
    def get_suggested_hyperparameters() -> Dict[str, Union[List[Any], Dict[str, Any]]]:
        """Return the tunable neural-network hyperparameters."""

        return {
            'normalizer': [norm.value for norm in NormalizerType],
            'sampler': [sample.value for sample in SamplerType],
            'hidden_layers': {'low': 1, 'high': 3, 'step': 1},
            'hidden_units': {'low': 64, 'high': 256, 'step': 32},
            'hidden_activation': ['tanh', 'relu', 'gelu'],
            'vsn': [True, False],
            'layer_normalization': [True, False],
            'batch_normalization': [True, False],
            'dropout_rate': {'low': 0.0, 'high': 0.5, 'step': 0.1},
            'odd_noise_std': {'low': 0.00, 'high': 0.2, 'step': 0.01},
            'class_weight': [True, False],
            'optimizer': ['adam', 'adabelief', 'adan', 'ranger25'],
            'lookahead': [True, False],
            'label_smoothing': {'low': 0.0, 'high': 0.1, 'step': 0.01},
            'learning_rate':  {'low': 0.0005, 'high': 0.05, 'step': 0.0005},
            'batch_size': {'low': 16, 'high': 128, 'step': 16},
            'lr_decay_patience': {'low': 0, 'high': 30, 'step': 5},
            'lr_decay_factor': {'low': 0.2, 'high': 0.5, 'step': 0.1}
        }

    def load(self, checkpoint_directory: str):
        """ Loads the classifier using tensorflow. """

        input_size = self._model_config['input_size']
        num_classes = self._model_config['num_classes']
        self._classifier = self.build(input_size=input_size, num_classes=num_classes)

        model_checkpoint = f'{checkpoint_directory}/model.ckpt'
        if os.path.exists(model_checkpoint) or os.path.exists(f'{model_checkpoint}.index'):
            self._classifier.model.load_weights(filepath=model_checkpoint)

        attention_checkpoint = f'{checkpoint_directory}/attn_model.ckpt'
        if (
                self._classifier.attn_model is not None
                and (
                    os.path.exists(attention_checkpoint)
                    or os.path.exists(f'{attention_checkpoint}.index')
                )
        ):
            self._classifier.attn_model.load_weights(filepath=attention_checkpoint)

    def save(self, checkpoint_directory: str):
        """ Saves the classifier using tensorflow. """

        if self._classifier is None:
            raise ValueError('Classifier has not been built/fit yet. Cannot save None classifier.')

        self._classifier.model.save_weights(filepath=f'{checkpoint_directory}/model.ckpt')

        if self._classifier.attn_model is not None:
            self._classifier.attn_model.save_weights(filepath=f'{checkpoint_directory}/attn_model.ckpt')

    def build(self, input_size: int, num_classes: int, seed: int = 0) -> TFModel:
        self._model_config['input_size'] = input_size
        self._model_config['num_classes'] = num_classes
        return TFModel(
            input_size=input_size,
            num_classes=num_classes,
            target_type=self._model_config['target_type'],
            hidden_layers=self._model_config['hidden_layers'],
            hidden_units=self._model_config['hidden_units'],
            hidden_activation=self._model_config['hidden_activation'],
            vsn=self._model_config['vsn'],
            layer_normalization=self._model_config['layer_normalization'],
            batch_normalization=self._model_config['batch_normalization'],
            dropout_rate=self._model_config['dropout_rate'],
            odd_noise_std=self._model_config['odd_noise_std'],
            class_weight=self._model_config['class_weight'],
            optimizer=self._model_config['optimizer'],
            lookahead=self._model_config['lookahead'],
            label_smoothing=self._model_config['label_smoothing'],
            learning_rate=self._model_config['learning_rate'],
            batch_size=self._model_config['batch_size'],
            epochs=self._model_config['epochs'],
            early_stopping_patience=self._model_config['early_stopping_patience'],
            lr_decay_patience=self._model_config['lr_decay_patience'],
            lr_decay_factor=self._model_config['lr_decay_factor'],
            verbose=self._model_config['verbose'],
            seed=seed
        )

    def fit(
            self,
            train_df: pd.DataFrame,
            eval_df: Optional[pd.DataFrame] = None,
            seed: Optional[int] = 0,
    ) -> pd.DataFrame:
        """ Fits the classifier in the data and returns the evaluation report.

            Args
            ----
            :param train_df: The train dataframe used for fitting.
            :param eval_df: The eval dataframe used for evaluation.
            :param seed: (Optional[int]) model's random seed, to enable result reproducibility.
        """

        if seed is None:
            seed = 0

        target_type = self._model_config['target_type']

        # Computing num classes.
        num_classes = len(target_type.get_mapper())
        one_hot = num_classes > 2

        # Preprocessing train-eval dataset.
        x_train, y_train, self._model_config['normalizer'] = self._dataset_preprocessor.preprocess(
            df=train_df,
            target_type=target_type,
            one_hot_targets=one_hot,
            normalizer=self._model_config['normalizer'],
            sampler=self._model_config['sampler'],
            seed=seed
        )

        if eval_df is None:
            if self._model_config['sampler'] is None:
                x_eval = x_train
                y_eval = y_train
            else:
                x_eval, y_eval, _ = self._dataset_preprocessor.preprocess(
                    df=train_df,
                    target_type=target_type,
                    one_hot_targets=one_hot,
                    normalizer=self._model_config['normalizer'],
                    sampler=None,
                    seed=seed
                )
        else:
            x_eval, y_eval, _ = self._dataset_preprocessor.preprocess(
                df=eval_df,
                target_type=target_type,
                one_hot_targets=one_hot,
                normalizer=self._model_config['normalizer'],
                sampler=None
            )

        # Computing class weights.
        if self._model_config['class_weight']:
            classes = np.array(list(target_type.get_mapper().values()), dtype=np.int32)
            y = y_train if num_classes == 2 else np.argmax(y_train, axis=1)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
            class_weights = dict(enumerate(weights))
        else:
            class_weights = None

        # Building classifier and adding CalibratedClassifierCV wrapper if specified.
        num_classes = len(target_type.get_mapper())
        self._classifier = self.build(input_size=x_train.shape[1], num_classes=num_classes, seed=seed)

        # Fitting classifier in extracted inputs, targets.
        self._classifier.fit(x=x_train, y=y_train, class_weights=class_weights, validation_data=(x_eval, y_eval))

        # Evaluating classifier.
        report, _ = self.evaluate(df=train_df, return_predictions=False)
        report.insert(loc=0, column='Data', value='Train')

        if eval_df is not None:
            eval_report, _ = self.evaluate(df=eval_df, return_predictions=False)
            eval_report.insert(loc=0, column='Data', value='Eval')
            report = pd.concat([report, eval_report], axis=0, ignore_index=True)

        return report
