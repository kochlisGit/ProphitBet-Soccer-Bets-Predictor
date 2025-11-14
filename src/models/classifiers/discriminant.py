from typing import Any, List, Dict, Optional, Union
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.covariance import OAS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from src.preprocessing.utils.sampling import SamplerType
from src.preprocessing.utils.target import TargetType
from src.models.model import ClassificationModel


class DiscriminantAnalysisClassifier(ClassificationModel):
    """ Implementation of Linear/Quadratic Discriminant Analysis (LDA) Classifier with Kernel PCA extension.

        Parameters
        -----------------------------

        :param league_id: The created league id.
        :param model_id: The provided model id by the user.
        :param target_type: The classification target type.
        :param normalizer: The input data normalizer placeholder.
        :param sampler: The sampler method for training data.
        :param oas: Whether to use OAS covariance estimator.
        :param decision_boundary: str. The decision boundary type. Supports "linear" or "quadratic".
    """

    def __init__(
            self,
            league_id: str,
            model_id: str,
            target_type: TargetType,
            normalizer: Optional[TransformerMixin] = None,
            sampler: Optional[SamplerType] = None,
            oas: bool = False,
            decision_boundary: str = 'linear',
            **kwargs
    ):
        self._oas = oas
        self._decision_boundary = decision_boundary

        super().__init__(
            league_id=league_id,
            model_id=model_id,
            target_type=target_type,
            calibrate_probabilities=False,
            normalizer=normalizer,
            sampler=sampler,
            **kwargs
        )

    @property
    def oas(self) -> bool:
        return self._oas

    @property
    def decision_boundary(self) -> str:
        return self._decision_boundary

    def build_classifier(self, input_size: int, num_classes: int) -> BaseEstimator:
        """ Builds a KNN classification model. """

        if self._decision_boundary == 'linear':
            if self._oas:
                oas = OAS(assume_centered=self._normalizer is None)
                clf = LinearDiscriminantAnalysis(n_components=2, solver='lsqr', covariance_estimator=oas)
            else:
                clf = LinearDiscriminantAnalysis(n_components=2)
        elif self._decision_boundary == 'quadratic':
            clf = QuadraticDiscriminantAnalysis()
        else:
            raise ValueError(f'Undefined decision boundary type: "{self._decision_boundary}"')

        return clf

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == 'oas':
            return [True, False]
        elif param == 'decision_boundary':
            return ['linear', 'quadratic']
        else:
            raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.update({
            'oas': self._oas,
            'decision_boundary': self._decision_boundary
        })
        return model_config
