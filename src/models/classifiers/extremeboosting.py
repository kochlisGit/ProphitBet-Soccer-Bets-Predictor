import numpy as np
from typing import Any, Dict, List, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from src.preprocessing.utils.sampling import SamplerType
from src.preprocessing.utils.target import TargetType
from src.models.model import ClassificationModel


class XGBoost(ClassificationModel):
    """ Implementation of Extreme Boosting Classifier
        (An improved version of Decision-Tree-Boosting method, that utilizes Gradient Descent).

        Parameters
        -----------------------------

        :param league_id: The created league id.
        :param model_id: The provided model id by the user.
        :param target_type: The classification target type.
        :param calibrate_probabilities: Whether to calibrate output probabilities.
        :param normalizer: The input data normalizer placeholder.
        :param sampler: The sampler method for training data.
        :param n_estimators: int. Number of decision tree classifiers.
        :param max_depth: int | None. The maximum tree dept. If None, there is no depth limit.
                                      Small depth values are ideal for interpretable trees.
        :param min_child_weight: int. Minimum weight of each child estimator.
        :param learning_rate: float. The learning_rate of the gradient descent step.
        :param lambda_regularization: float. The lambda regularization value.
        :param alpha_regularization: float. The alpha regularization value.
    """

    def __init__(
            self,
            league_id: str,
            model_id: str,
            target_type: TargetType,
            calibrate_probabilities: bool,
            normalizer: Optional[TransformerMixin] = None,
            sampler: Optional[SamplerType] = None,
            n_estimators: int = 100,
            max_depth: int = 6,
            min_child_weight: int = 1,
            learning_rate: float = 0.3,
            lambda_regularization: float = 1.0,
            alpha_regularization: float = 0.0,
            **kwargs
    ):
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._min_child_weight = min_child_weight
        self._learning_rate = learning_rate
        self._lambda_regularization = lambda_regularization
        self._alpha_regularization = alpha_regularization

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
        """ Builds an XGBoost classification model. """

        return XGBClassifier(
            booster='gbtree',
            n_estimators=self._n_estimators,
            learning_rate=self._learning_rate,
            max_depth=self._max_depth,
            min_child_weight=self._min_child_weight,
            reg_lambda=self._lambda_regularization,
            reg_alpha=self._alpha_regularization,
            random_state=0,
            n_jobs=-1
        )

    def get_feature_importances(self) -> np.ndarray:
        """ Gets feature importances of XGBoost. """

        if self._calibrate_probabilities:
            return self._classifier.calibrated_classifiers_[0].estimator.feature_importances_
        else:
            return self._classifier.feature_importances_

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == 'n_estimators':
            return {'low': 50, 'high': 500, 'step': 50}
        if param == 'max_depth':
            return {'low': 1, 'high': 15, 'step': 1}
        elif param == 'min_child_weight':
            return {'low': 1, 'high': 5, 'step': 1}
        elif param == 'learning_rate':
            return {'low': 0.005, 'high': 0.5, 'step': 0.005}
        elif param == 'lambda_regularization':
            return {'low': 0.1, 'high': 2.0, 'step': 0.1}
        elif param == 'alpha_regularization':
            return {'low': 0.0, 'high': 1.0, 'step': 0.1}
        else:
            raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.update({
            'n_estimators': self._n_estimators,
            'max_depth': self._max_depth,
            'min_child_weight': self._min_child_weight,
            'learning_rate': self._learning_rate,
            'lambda_regularization': self._lambda_regularization,
            'alpha_regularization': self._alpha_regularization
        })
        return model_config
