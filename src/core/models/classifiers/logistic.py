import numpy as np
from typing import Any, Dict, List, Optional, Union
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from src.core.preprocessing.utils import NormalizerType, SamplerType, TargetType
from src.core.models.model import Classifier


class LogisticRegressor(Classifier):
    """ Implementation of the Logistic Regression Classifier.

       Attributes
       -----------
        :param target_type: (TargetType) classification target type.
        :param normalizer: (str | NormalizerType | TransformMixIn) input data normalizer placeholder. If NormalizerType provided,
                           then the normalizer is built and stored once fit method is called.
        :param sampler: (str | SamplerType | BaseSampler) sampler method for training data.
        :param calibrate_probabilities: (bool) whether to calibrate output probabilities.
        :param penalty: (Optional[str])  penalty method, which can either be "l1", "l2", "elastic" --> (l1 + l2) or None.
                                    l1: minimizes (WX - y)^2 - |W|. l2 minimizes (WX - y)^2 - W^2. elastic uses both
                                    l1 and l2. None does not use a regularization term (minimizes (WX - Y)^2).
   """

    def __init__(
            self,
            target_type: TargetType,
            normalizer: Optional[Union[NormalizerType, TransformerMixin]] = None,
            sampler: Optional[SamplerType] = None,
            calibrate_probabilities: bool = True,
            penalty: Optional[str] = None,
            **kwargs
    ):
        self._cs = [0.01, 0.1, 1.0, 5.0, 10]
        self._blend_factors = [0.01, 0.1, 0.2, 0.5, 1.0]

        super().__init__(
            target_type=target_type,
            normalizer=normalizer,
            sampler=sampler,
            calibrate_probabilities=calibrate_probabilities,
            n_jobs=None,
            penalty=penalty
        )

    @staticmethod
    def get_suggested_hyperparameters() -> Dict[str, Union[List[Any], Dict[str, Any]]]:
        """ Returns a suggested set of all hyperparameters (Dict[str, values]) for the classifier.  """

        return {
            'normalizer': [norm.value for norm in NormalizerType],
            'sampler': [sample.value for sample in SamplerType],
            'calibrate_probabilities': [True, False],
            'penalty': [None, 'l1', 'l2', 'elastic']
        }

    def build(self, input_size: int, num_classes: int, seed: int = 0) -> LogisticRegression:
        penalty = self._model_config['penalty']

        if penalty is None:
            return LogisticRegression(solver='lbfgs', max_iter=5000)
        elif penalty == 'l2':
            return LogisticRegressionCV(
                Cs=self._cs,
                l1_ratios=(0.0,),
                solver='lbfgs',
                scoring='accuracy',
                max_iter=5000,
                random_state=seed,
                n_jobs=-1,
                use_legacy_attributes=False
            )
        elif penalty == 'l1':
            return LogisticRegressionCV(
                Cs=self._cs,
                l1_ratios=(1.0,),
                solver='saga',
                scoring='accuracy',
                max_iter=5000,
                random_state=seed,
                n_jobs=-1,
                use_legacy_attributes=False
            )
        elif penalty == 'elastic':
            return LogisticRegressionCV(
                Cs=self._cs,
                l1_ratios=self._blend_factors,
                solver='saga',
                scoring='accuracy',
                max_iter=5000,
                random_state=seed,
                n_jobs=-1,
                use_legacy_attributes=False
            )
        else:
            raise ValueError(f'Undefined penalty type: "{penalty}".')

    def get_coefficients(self) -> np.ndarray:
        """ Returns the coefficients of the Logistic Regression model. """

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
