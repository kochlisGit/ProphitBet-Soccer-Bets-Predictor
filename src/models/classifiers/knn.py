from typing import Any, Dict, List, Optional, Union
from sklearn.base import TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from src.preprocessing.utils.sampling import SamplerType
from src.preprocessing.utils.target import TargetType
from src.models.model import ClassificationModel


class KNN(ClassificationModel):
    """ Implementation of K-Nearest Neighbors Classifier.

        Parameters
        -----------------------------

        :param league_id: The created league id.
        :param model_id: The provided model id by the user.
        :param target_type: The classification target type.
        :param calibrate_probabilities: Whether to calibrate output probabilities.
        :param normalizer: The input data normalizer placeholder.
        :param sampler: The sampler method for training data.
        :param n_neighbors: int. The number of decision neighbors. The class of each instance is decided based on the
                                 majority class of the N closest instances. Usually, Low number of neighbors can capture
                                 complex patterns.
        :param weights: str. The weight of each closest neighbor. It can be either "uniform" or "distance".
        :param p: int. The distance type (set p=1 for Manhattan and p=2 for Euclidean).
    """

    def __init__(
            self,
            league_id: str,
            model_id: str,
            target_type: TargetType,
            calibrate_probabilities: bool,
            normalizer: Optional[TransformerMixin] = None,
            sampler: Optional[SamplerType] = None,
            n_neighbors: int = 5,
            weights: str = 'uniform',
            p: int = 2,
            **kwargs
    ):
        self._n_neighbors = n_neighbors
        self._weights = weights
        self._p = p

        super().__init__(
            league_id=league_id,
            model_id=model_id,
            target_type=target_type,
            calibrate_probabilities=calibrate_probabilities,
            normalizer=normalizer,
            sampler=sampler,
            **kwargs
        )

    @property
    def n_neighbors(self) -> int:
        return self._n_neighbors

    @property
    def weights(self) -> str:
        return self._weights

    @property
    def p(self) -> int:
        return self._p

    def build_classifier(self, input_size: int, num_classes: int) -> KNeighborsClassifier:
        """ Builds a KNN classification model. """

        return KNeighborsClassifier(n_neighbors=self._n_neighbors, weights=self._weights, p=self._p, n_jobs=-1)

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == 'n_neighbors':
            return {'low': 3, 'high': 99, 'step': 6}
        elif param == 'weights':
            return ['uniform', 'distance']
        elif param == 'p':
            return [1, 2]
        else:
            raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.update({
            'n_neighbors': self._n_neighbors,
            'weights': self._weights,
            'p': self._p
        })
        return model_config
