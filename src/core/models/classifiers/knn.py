from typing import Any, Dict, List, Optional, Union
from sklearn.base import TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from src.core.preprocessing.utils import NormalizerType, SamplerType, TargetType
from src.core.models.model import Classifier


class KNearestNeighbors(Classifier):
    """ Implementation of the K-Nearest Neighbor classifier.

       Attributes
       -----------
        :param target_type: (TargetType_ classification target type.

        :param normalizer: (str | NormalizerType | TransformMixIn) input data normalizer placeholder. If NormalizerType provided,
                           then the normalizer is built and stored once fit method is called.
        :param sampler: (str | SamplerType | BaseSampler) sampler method for training data.
        :param calibrate_probabilities: (bool) whether to calibrate output probabilities.
        :param n_neighbors: (int) number of decision neighbors. The class of each instance is decided based on the
                                 majority class of the N closest instances. Usually, Low number of neighbors can capture
                                 complex patterns.
        :param weights: (str) weight of each closest neighbor. It can be either "uniform" or "distance".
        :param p: (int) distance type (set p=1 for Manhattan and p=2 for Euclidean).
   """

    def __init__(
            self,
            target_type: TargetType,
            normalizer: Optional[Union[NormalizerType, TransformerMixin]] = None,
            sampler: Optional[SamplerType] = None,
            calibrate_probabilities: bool = True,
            n_neighbors: int = 5,
            weights: str = 'uniform',
            p: int = 2,
            **kwargs
    ):
        super().__init__(
            target_type=target_type,
            normalizer=normalizer,
            sampler=sampler,
            calibrate_probabilities=calibrate_probabilities,
            n_jobs=-1,
            n_neighbors=n_neighbors,
            weights=weights,
            p=p
        )

    @staticmethod
    def get_suggested_hyperparameters() -> Dict[str, Union[List[Any], Dict[str, Any]]]:
        """ Returns a suggested set of all hyperparameters (Dict[str, values]) for the classifier.  """

        return {
            'normalizer': [norm.value for norm in NormalizerType],
            'sampler': [sample.value for sample in SamplerType],
            'calibrate_probabilities': [True, False],
            'n_neighbors': {'low': 3, 'high': 151, 'step': 2},
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }

    def build(self, input_size: int, num_classes: int, seed: int = 0) -> KNeighborsClassifier:
        return KNeighborsClassifier(
            n_neighbors=self._model_config['n_neighbors'],
            weights=self._model_config['weights'],
            p=self._model_config['p'],
            n_jobs=-1
        )
