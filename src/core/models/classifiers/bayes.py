from typing import Any, Dict, List, Optional, Union
from sklearn.base import TransformerMixin
from sklearn.naive_bayes import GaussianNB
from src.core.preprocessing.utils import NormalizerType, SamplerType, TargetType
from src.core.models.model import Classifier


class NaiveBayes(Classifier):
    """ Implementation of the Gaussian Naive Bayes classifier.

       Attributes
       -----------
        :param target_type: (TargetType) classification target type.
        :param normalizer: (str | NormalizerType | TransformMixIn) input data normalizer placeholder. If NormalizerType provided,
                           then the normalizer is built and stored once fit method is called.
        :param sampler: (str | SamplerType | BaseSampler) sampler method for training data.
        :param calibrate_probabilities: (bool) whether to calibrate output probabilities.
   """

    def __init__(
            self,
            target_type: TargetType,
            normalizer: Optional[Union[NormalizerType, TransformerMixin]] = None,
            sampler: Optional[SamplerType] = None,
            calibrate_probabilities: bool = True,
            **kwargs
    ):
        super().__init__(
            target_type=target_type,
            normalizer=normalizer,
            sampler=sampler,
            calibrate_probabilities=calibrate_probabilities,
            n_jobs=None
        )

    @staticmethod
    def get_suggested_hyperparameters() -> Dict[str, Union[List[Any], Dict[str, Any]]]:
        """ Returns a suggested set of all hyperparameters (Dict[str, values]) for the classifier.  """

        return {
            'normalizer': [norm.value for norm in NormalizerType],
            'sampler': [sample.value for sample in SamplerType],
            'calibrate_probabilities': [True, False],
        }

    def build(self, input_size: int, num_classes: int, seed: int = 0) -> GaussianNB:
        return GaussianNB()
