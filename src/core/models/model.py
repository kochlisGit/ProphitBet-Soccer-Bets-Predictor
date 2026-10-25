import pickle
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from src.core.metrics.utils import generate_report, ODDS_COLUMNS
from src.core.preprocessing.dataset import DatasetPreprocessor
from src.core.preprocessing.filters import EvalFilter
from src.core.preprocessing.utils import NormalizerType, SamplerType, TargetType


class Classifier(ABC):
    """ Base class wrapper for Scikit-Learn-based classification models.

        Properties
        -----------

        * All models inside this class are treated as Scikit-Learn classifiers.
        * Custom classifiers should implement BaseEstimator & TranformerMixin class.
        * Custom classifiers should be serializable (to enable save/load as pickle objects).

        Attributes
        ----------
        :param target_type: (TargetType) classification target type.
        :param normalizer: (str | NormalizerType | TransformMixIn) input data normalizer placeholder. If
                           NormalizerType provided, then the normalizer is built and stored once fit method is called.
        :param sampler: (str | SamplerType | BaseSampler) sampler method for training data.
        :param calibrate_probabilities: (bool) whether to calibrate output probabilities.
        :param n_jobs: (int) number of parallel jobs.
        :param input_size: (int | None) number of model inputs after the classifier is built.
        :param num_classes: (int | None) number of model outputs after the classifier is built.
    """

    def __init__(
            self,
            target_type: TargetType,
            normalizer: Optional[Union[str, NormalizerType, TransformerMixin]] = None,
            sampler: Optional[SamplerType] = None,
            calibrate_probabilities: bool = True,
            n_jobs: Optional[int] = None,
            input_size: Optional[int] = None,
            num_classes: Optional[int] = None,
            **kwargs
    ):
        self._model_config = {
            'model_cls': self.__class__,
            'target_type': target_type,
            'normalizer': normalizer,
            'sampler': sampler,
            'calibrate_probabilities': calibrate_probabilities,
            'n_jobs': n_jobs,
            'input_size': input_size,
            'num_classes': num_classes,
            **kwargs
        }
        self._dataset_preprocessor = DatasetPreprocessor()
        self._classifier = None
        self._decimals = 3

    @property
    def model_config(self) -> Dict[str, Any]:
        return self._model_config

    @property
    def classifier(self) -> Optional[BaseEstimator]:
        return self._classifier

    @staticmethod
    def from_config(model_config: Dict[str, Any]) -> 'Classifier':
        """ Reconstructs a classifier wrapper from a stored model configuration. """

        config = model_config.copy()
        model_cls = config.pop('model_cls')
        return model_cls(**config)

    @abstractmethod
    def build(self, input_size: int, num_classes: int, seed: int = 0) -> BaseEstimator:
        """ Builds a selected classifier. It should be a BaseEstimator instance.

            :param input_size: (int) model's input size. It is typically required to build the model's architecture.
            :param num_classes: (int) model's output size. It is typically required to build the model's architecture.
            :param seed: (int) model's random seed, to enable result reproducibility.
        """

        pass

    @staticmethod
    @abstractmethod
    def get_suggested_hyperparameters() -> Dict[str, Union[List[Any], Dict[str, Any]]]:
        """ Returns a suggested set of all hyperparameters (Dict[str, values]) for the classifier.  """

        pass

    def load(self, checkpoint_directory: str):
        """ Loads the classifier using pickle. Override for custom loading method.

            Args
            ----
            :param checkpoint_directory: (str) filepath of the checkpoint directory.
        """

        with open(f'{checkpoint_directory}/classifier.pkl', mode='rb') as pklfile:
            self._classifier = pickle.load(file=pklfile)

    def save(self, checkpoint_directory: str):
        """ Saves the classifier using pickle into the specified location. Override for custom saving method.

            Args
            ----
            :param checkpoint_directory: (str) filepath of the checkpoint directory.
        """

        if self._classifier is None:
            raise ValueError('Classifier has not been built/fit yet. Cannot save None classifier.')

        with open(f'{checkpoint_directory}/classifier.pkl', mode='wb') as pklfile:
            pickle.dump(obj=self._classifier, file=pklfile)

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
            :param seed: (int | None) model's random seed, to enable result reproducibility.
        """

        if seed is None:
            seed = 0

        # Preprocessing train dataset.
        target_type = self._model_config['target_type']
        x, y, self._model_config['normalizer'] = self._dataset_preprocessor.preprocess(
            df=train_df,
            target_type=target_type,
            normalizer=self._model_config['normalizer'],
            sampler=self._model_config['sampler'],
            seed=seed
        )

        # Building classifier and adding CalibratedClassifierCV wrapper if specified.
        input_size = x.shape[1]
        num_classes = len(target_type.get_mapper())
        self._model_config['input_size'] = input_size
        self._model_config['num_classes'] = num_classes
        clf = self.build(input_size=input_size, num_classes=num_classes, seed=seed)

        # Wrapping classifier with calibrated classifier.
        if self._model_config['calibrate_probabilities']:
            n_jobs = self._model_config['n_jobs']
            calib_jobs = -1 if n_jobs is None or n_jobs == 1 else 1
            self._classifier = CalibratedClassifierCV(estimator=clf, method='isotonic', n_jobs=calib_jobs)
        else:
            self._classifier = clf

        # Fitting classifier in extracted inputs, targets.
        self._classifier.fit(x, y)

        # Evaluating classifier.
        report, _ = self.evaluate(df=train_df, return_predictions=False)
        report.insert(loc=0, column='Data', value='Train')

        if eval_df is not None:
            eval_report, _ = self.evaluate(df=eval_df, return_predictions=False)
            eval_report.insert(loc=0, column='Data', value='Eval')
            report = pd.concat([report, eval_report], axis=0, ignore_index=True)

        return report

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """ Generates the predicted results for a given dataframe.

            Args
            ----
            :param df: (pd.DataFrame) league's dataframe (without preprocessing).
        """

        if self._classifier is None:
            raise RuntimeError('Classifier has not been built/fit yet.')

        # Extracting inputs, targets.
        x, _ = self._dataset_preprocessor.construct_inputs(df=df, normalizer=self._model_config['normalizer'])
        y_pred = self._classifier.predict(x)
        return y_pred

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """ Generates the predicted probabilities for a given dataframe.

            Args
            ----
            :param df: (pd.DataFrame) league's dataframe (without preprocessing).
        """

        if self._classifier is None:
            raise RuntimeError('Classifier has not been built/fit yet.')

        # Extracting inputs, targets.
        x, _ = self._dataset_preprocessor.construct_inputs(df=df, normalizer=self._model_config['normalizer'])
        y_pred = self._classifier.predict_proba(x)
        return y_pred

    def evaluate(self, df: pd.DataFrame, return_predictions: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """ Computes classification, performance and skill metrics (for "Result" target only).

            Args
            ----
            :param df: (pd.DataFrame) league dataframe (matches) to be evaluated.
            :param return_predictions: (bool) whether to return model predictions along with the probabilities.
        """

        # Generating inputs, targets, probabilities.
        x, y_true, _ = self._dataset_preprocessor.preprocess(
            df=df,
            target_type=self._model_config['target_type'],
            normalizer=self._model_config['normalizer']
        )
        y_prob = self._classifier.predict_proba(x)
        y_pred = np.argmax(y_prob, axis=1)
        odds = df[ODDS_COLUMNS]
        start_date = df['Date'].iloc[df.shape[0] - 1]
        end_date = df['Date'].iloc[0]

        # Generating report dataframe.
        report = generate_report(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            odds=odds,
            target_type=self._model_config['target_type'],
            start_date=start_date,
            end_date=end_date
        )

        # Constructing prediction dataframe.
        if return_predictions:
            predictions_df = pd.DataFrame({
                'Predicted': y_pred,
                **{f'Prob-{i}': y_prob[:, i] for i in range(y_prob.shape[1])}
            })
        else:
            predictions_df = None

        return report, predictions_df


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """ Model's config class that stores classifier parameters and model reports.

        Attributes
        ----------
        :param league_id: (str) user-specified league identifier.
        :param model_id: (str) user-specified model identifier for the selected league.
        :param classifier_config: (Dict[str, Any]) parameters required to reconstruct the classifier.
        :param num_train_samples: (int) number of samples used for training.
        :param num_eval_samples: (int) number of samples used for evaluation.
        :param train_report: pd.DataFrame model's training metrics.
        :param cv_report: (pd.DataFrame | None) model's cross-validation metrics.
        :param sliding_cv_report: (pd.DataFrame | None) model's sliding cross-validation metrics.
        :param tune_report: (pd.DataFrame | None) whether hyperparameter optimization has been applied.
        :param nested_cv_tune_report: (pd.DataFrame | None) whether hyperparameter optimization with nested cv been applied.
        :param tune_param_importances: (Dict[str, float] | None) parameter importances extracted during the tuning process.
        :param eval_filters: saved evaluation filters, including their prediction mode.
    """

    league_id: str
    model_id: str
    classifier_config: Dict[str, Any]
    num_train_samples: int
    num_eval_samples: int
    train_report: pd.DataFrame
    cv_report: Optional[pd.DataFrame] = None
    sliding_cv_report: Optional[pd.DataFrame] = None
    tune_report: Optional[pd.DataFrame] = None
    nested_cv_tune_report: Optional[pd.DataFrame] = None
    tune_param_importances: Optional[Dict[str, float]] = None
    eval_filters: Optional[List[EvalFilter]] = None
