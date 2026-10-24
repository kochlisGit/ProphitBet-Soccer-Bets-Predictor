import numpy as np
from typing import Any, Dict, List, Optional, Union
from sklearn.base import TransformerMixin
from sklearn.svm import SVC, LinearSVC
from src.core.preprocessing.utils import NormalizerType, SamplerType, TargetType
from src.core.models.model import Classifier


class SupportVectorMachine(Classifier):
    """ Implementation of the Support Vector Machine classifier.

       Attributes
       -----------
        :param target_type: (TargetType) classification target type.
        :param normalizer: (str | NormalizerType | TransformMixIn) input data normalizer placeholder. If NormalizerType provided,
                           then the normalizer is built and stored once fit method is called.
        :param sampler: (str | SamplerType | BaseSampler) sampler method for training data.
        :param kernel: (str | None) kernel function of SVM. Supported kernels are: None, "linear", "rbf", "poly" or "sigmoid".
        :param degree: (int) polynomial degree of kernel function (if kernel == "poly").
        :param gamma: (int) gamma regularization coefficient (if kernel == "poly").
        :param class_weight: (bool) whether to assign different weights to minority-majority classes (good for classification tasks with imbalances).
   """

    def __init__(
            self,
            target_type: TargetType,
            normalizer: Optional[Union[NormalizerType, TransformerMixin]] = None,
            sampler: Optional[SamplerType] = None,
            kernel: Optional[str] = 'rbf',
            degree: int = 3,
            gamma: float = 1.0,
            class_weight: bool = True,
            **kwargs
    ):
        super().__init__(
            target_type=target_type,
            normalizer=normalizer,
            sampler=sampler,
            calibrate_probabilities=True,
            n_jobs=None,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            class_weight=class_weight
        )

    @staticmethod
    def get_suggested_hyperparameters() -> Dict[str, Union[List[Any], Dict[str, Any]]]:
        """ Returns a suggested set of all hyperparameters (Dict[str, values]) for the classifier.  """

        return {
            'normalizer': [norm.value for norm in NormalizerType],
            'sampler': [sample.value for sample in SamplerType],
            'kernel': [None, 'linear', 'rbf', 'poly', 'sigmoid'],
            'degree': {'low': 3, 'high': 6, 'step': 1},
            'gamma': {'low': 0.1, 'high': 2.0, 'step': 0.1},
            'class_weight': [True, False]
        }

    def build(self, input_size: int, num_classes: int, seed: int = 0) -> Union[SVC, LinearSVC]:
        class_weight = 'balanced' if self._model_config['class_weight'] else None

        if self._model_config['kernel'] is None:
            return LinearSVC(
                class_weight=class_weight,
                random_state=seed,
            )

        return SVC(
            C=self._model_config['gamma'],
            degree=self._model_config['degree'],
            kernel=self._model_config['kernel'],
            class_weight=class_weight,
            random_state=seed
        )

    def get_coefficients(self) -> np.ndarray:
        """ Returns the coefficients of the SVM model. """

        if self._model_config['calibrate_probabilities']:
            coeffs = np.stack([
                calibrated.estimator.coef_
                for calibrated in self._classifier.calibrated_classifiers_
            ])
            values = np.mean(np.abs(coeffs), axis=0)
        else:
            values = np.abs(self._classifier.coef_)

        num_classes=len(self._model_config['target_type'].get_mapper())
        return values if num_classes > 2 else values[0]
