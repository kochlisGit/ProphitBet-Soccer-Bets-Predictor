from typing import Any, Dict, List, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from src.preprocessing.utils.sampling import SamplerType
from src.preprocessing.utils.target import TargetType
from src.models.model import ClassificationModel


class NaiveBayes(ClassificationModel):
    """ Implementation of Naive Bayes Classifier.

        Parameters
        -----------------------------

        :param league_id: The created league id.
        :param model_id: The provided model id by the user.
        :param target_type: The classification target type.
        :param calibrate_probabilities: Whether to calibrate output probabilities.
        :param normalizer: The input data normalizer placeholder.
        :param sampler: The sampler method for training data.
        :param algorithm: str. The Naive-Bayes algorithm. It can either be "gaussian", "multinomial" or "complement".
                               Gaussian assumes that each feature follows a Normal distribution. Multinomial assumes
                               that each feature follows a Multinomial distribution. Complement is good for
                               imbalanced classes.
    """

    def __init__(
            self,
            league_id: str,
            model_id: str,
            target_type: TargetType,
            calibrate_probabilities: bool,
            normalizer: Optional[TransformerMixin] = None,
            sampler: Optional[SamplerType] = None,
            algorithm: str = 'gaussian',
            **kwargs
    ):
        self._algorithm = algorithm

        super().__init__(
            league_id=league_id,
            model_id=model_id,
            target_type=target_type,
            calibrate_probabilities=calibrate_probabilities,
            normalizer=normalizer,
            sampler=sampler,
            **kwargs
        )

    def build_classifier(self, input_size: int, num_classes: int) -> BaseEstimator:
        """ Builds a Naive-Bayes classification model. """

        if self._algorithm == 'gaussian':
            return GaussianNB()
        elif self._algorithm == 'multinomial':
            return MultinomialNB()
        elif self._algorithm == 'complement':
            return ComplementNB()
        else:
            raise ValueError(f'Undefined Naive Bayes algorithm: "{self._algorithm}".')

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == 'algorithm':
            return ['gaussian', 'multinomial', 'complement']
        else:
            raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.update({
            'algorithm': self._algorithm
        })
        return model_config
