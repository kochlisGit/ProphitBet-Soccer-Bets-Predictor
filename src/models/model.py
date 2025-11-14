import pickle
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.preprocessing.dataset import DatasetPreprocessor
from src.preprocessing.utils.normalization import NormalizerType
from src.preprocessing.utils.sampling import SamplerType
from src.preprocessing.utils.target import TargetType


class ClassificationModel(ABC):
    """ Base class wrapper for Scikit-Learn-based classification models.

        Properties:
        ----------------------------------------------------------------------------

        * All models inside this class are treated as Scikit-Learn classifiers.
        * Custom classifiers should implement BaseEstimator & TranformerMixin class.
        * Custom classifiers should be serializable (to enable save/load as pickle objects).

        Parameters
        -----------------------------------------------------------------------------

        :param league_id: The created league id.
        :param model_id: The provided model id by the user.
        :param target_type: The classification target type.
        :param normalizer: The input data normalizer placeholder. If NormalizerType provided,
                           then the normalizer is built and stored once fit method is called.
        :param sampler: The sampler method for training data.
        :param calibrate_probabilities: bool. Whether to calibrate output probabilities.
    """

    def __init__(
            self,
            league_id: str,
            model_id: str,
            target_type: TargetType,
            normalizer: Optional[Union[NormalizerType, TransformerMixin]] = None,
            sampler: Optional[SamplerType] = None,
            calibrate_probabilities: bool = True,
            **kwargs
    ):
        self._league_id = league_id
        self._model_id = model_id
        self._target_type = target_type
        self._normalizer = normalizer
        self._sampler = sampler
        self._calibrate_probabilities = calibrate_probabilities

        self._dataset_preprocessor = DatasetPreprocessor()
        self._classifier = None

        self._precision = 3

    @property
    def league_id(self) -> str:
        return self._league_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def target_type(self) -> TargetType:
        return self._target_type

    @property
    def normalizer(self) -> TransformerMixin:
        return self._normalizer

    @property
    def calibrate_probabilities(self) -> bool:
        return self._calibrate_probabilities

    @property
    def classifier(self) -> BaseEstimator:
        return self._classifier

    @abstractmethod
    def build_classifier(self, input_size: int, num_classes: int) -> BaseEstimator:
        """ Builds a selected classifier. It should be a BaseEstimator instance. """

        pass

    @classmethod
    @abstractmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        pass

    @abstractmethod
    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @classmethod
    def get_suggest_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == 'calibrate_probabilities':
            return [True, False]
        elif param == 'normalizer':
            return [NormalizerType.STANDARD, NormalizerType.MIN_MAX, NormalizerType.MAX_ABS]
        elif param == 'sampler':
            return [SamplerType.SVM_SMOTE, SamplerType.NEARMISS, SamplerType.INSTANCE_HARDNESS_THRESHOLD]
        else:
            return cls._get_suggested_param_values(param=param)

    def get_default_model_config(self) -> Dict[str, Any]:
        base_config = {
            'cls': self.__class__,
            'league_id': self._league_id,
            'model_id': self._model_id,
            'target_type': self._target_type,
            'normalizer': self._normalizer,
            'sampler': self._sampler,
            'calibrate_probabilities': self._calibrate_probabilities
        }
        return self._get_model_config(model_config=base_config)

    def load(self, checkpoint_directory: str):
        """ Loads the classifier using pickle. """

        with open(f'{checkpoint_directory}/classifier.pkl', mode='rb') as pklfile:
            self._classifier = pickle.load(file=pklfile)

    def save(self, checkpoint_directory: str):
        """ Saves the classifier using pickle. """

        if self._classifier is None:
            raise ValueError('Classifier has not been built/fit yet. Cannot save None classifier.')

        with open(f'{checkpoint_directory}/classifier.pkl', mode='wb') as pklfile:
            pickle.dump(obj=self._classifier, file=pklfile)

    def fit(self, train_df: pd.DataFrame, eval_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """ Fits the classifier in the data and returns the evaluation report.
            :param train_df: The train dataframe used for fitting.
            :param eval_df: The eval dataframe used for evaluation.
        """

        x, y, self._normalizer = self._dataset_preprocessor.preprocess_dataset(
            df=train_df,
            target_type=self._target_type,
            normalizer=self._normalizer,
            sampler=self._sampler,
            seed=0
        )

        # Build & Train classifier.
        clf = self.build_classifier(input_size=x.shape[1], num_classes=np.unique(y).shape[0])

        if self._calibrate_probabilities:
            clf = CalibratedClassifierCV(estimator=clf, method='isotonic', n_jobs=-1)

        self._classifier = clf
        self._classifier.fit(x, y)

        # Evaluate classifier.
        if eval_df is not None:
            self._num_eval_samples = eval_df.shape[0]

        metrics_df = self._evaluate_classifier(train_df=train_df, eval_df=eval_df)
        return metrics_df

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Evaluates the classifier. """

        y_pred, y_true = self.predict(df=df, return_targets=True)
        return self.compute_metrics(y_true=y_true, y_pred=y_pred)

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        if y_true.shape[0] == 0:
            return pd.DataFrame({'Accuracy': [0.0], 'F1': [0.0], 'Precision': [0.0], 'Recall': [0.0]})

        return pd.DataFrame({
            'Accuracy': [round(accuracy_score(y_true=y_true, y_pred=y_pred), self._precision)],
            'F1': [round(f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0.0), self._precision)],
            'Precision': [round(precision_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0.0), self._precision)],
            'Recall': [round(recall_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0.0), self._precision)]
        })

    def predict(self, df: pd.DataFrame, return_targets: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """ Generates the predictions from a dataframe (and optionally returns the targets). """

        if self._classifier is None:
            raise RuntimeError('Classifier has not been built/fit yet.')

        x, y, _ = self._dataset_preprocessor.preprocess_dataset(
            df=df,
            target_type=self._target_type,
            normalizer=self._normalizer
        )

        y_pred = self._classifier.predict(x)
        y_true = y if return_targets else None
        return y_pred, y_true

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """ Generates the output probabilities for each label from a dataframe. """

        if self._classifier is None:
            raise RuntimeError('Classifier has not been built/fit yet.')

        x, _, _ = self._dataset_preprocessor.preprocess_dataset(
            df=df,
            target_type=self._target_type,
            normalizer=self._normalizer
        )
        return self._classifier.predict_proba(x)

    def _evaluate_classifier(self, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
        """ Evaluate classifier on train/eval sets."""

        metrics_df = self.evaluate(df=train_df)
        metrics_df['data'] = 'train'

        if eval_df is not None:
            eval_metrics_df = self.evaluate(df=eval_df)
            eval_metrics_df['data'] = 'eval'
            metrics_df = pd.concat([metrics_df, eval_metrics_df], ignore_index=True, axis=0)

        return metrics_df
