import numpy as np
from typing import Any, Dict, List, Optional, Union
from sklearn.base import TransformerMixin
from sklearn.covariance import OAS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from src.core.preprocessing.utils import NormalizerType, SamplerType, TargetType
from src.core.models.model import Classifier


class DiscriminantAnalysis(Classifier):
    """ Implementation of the Linear or Quadratic Discriminant Analysis classifier.

       Attributes
       -----------
        :param target_type: (TargetType) classification target type.
        :param normalizer: (str | NormalizerType | TransformMixIn) input data normalizer placeholder. If NormalizerType provided,
                           then the normalizer is built and stored once fit method is called.
        :param sampler: (str | SamplerType | BaseSampler) sampler method for training data.
        :param calibrate_probabilities: (bool) whether to calibrate output probabilities.
        :param oas: (bool) whether to use OAS covariance estimator.
        :param decision_boundary: (str) decision boundary type. Supported boundaries: "linear" or "quadratic".
        :param reg_param (float): regularization parameter for QuadraticDiscriminantAnalysis.
   """

    def __init__(
            self,
            target_type: TargetType,
            normalizer: Optional[Union[NormalizerType, TransformerMixin]] = None,
            sampler: Optional[SamplerType] = None,
            calibrate_probabilities: bool = True,
            oas: bool = False,
            decision_boundary: str = 'linear',
            reg_param: float = 0.001,
            **kwargs
    ):
        super().__init__(
            target_type=target_type,
            normalizer=normalizer,
            sampler=sampler,
            calibrate_probabilities=calibrate_probabilities,
            n_jobs=None,
            oas=oas,
            decision_boundary=decision_boundary,
            reg_param=reg_param
        )

    @staticmethod
    def get_suggested_hyperparameters() -> Dict[str, Union[List[Any], Dict[str, Any]]]:
        """ Returns a suggested set of all hyperparameters (Dict[str, values]) for the classifier.  """

        return {
            'normalizer': [norm.value for norm in NormalizerType],
            'sampler': [sample.value for sample in SamplerType],
            'calibrate_probabilities': [True, False],
            'oas': [True, False],
            'decision_boundary': ['linear', 'quadratic'],
            'reg_param': {'low': 0.001, 'high': 0.1, 'step': 0.01}
        }

    def build(
            self,
            input_size: int,
            num_classes: int,
            seed: int = 0,
    ) -> Union[LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis]:
        n_components = num_classes - 1
        decision_boundary = self._model_config['decision_boundary']

        if decision_boundary == 'linear':
            if self._model_config['oas']:
                oas = OAS(assume_centered=self._model_config['normalizer'] is None)
                clf = LinearDiscriminantAnalysis(n_components=n_components, solver='lsqr', covariance_estimator=oas)
            else:
                clf = LinearDiscriminantAnalysis(n_components=n_components)
        elif decision_boundary == 'quadratic':
            clf = QuadraticDiscriminantAnalysis(reg_param=self._model_config['reg_param'])
        else:
            raise ValueError(f'Undefined decision boundary type: "{decision_boundary}".')

        return clf

    def get_coefficients(self) -> Optional[np.ndarray]:
        """ Returns the coefficients of the Logistic Regression model. """

        if self._model_config['decision_boundary'] != 'linear':
            return None

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
