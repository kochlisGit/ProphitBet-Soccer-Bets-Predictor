import numpy as np
from typing import Any, Dict, List, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing.utils.sampling import SamplerType
from src.preprocessing.utils.target import TargetType
from src.models.model import ClassificationModel


class RandomForest(ClassificationModel):
    """ Implementation of Random Forest Classifier (An ensemble method that uses decision trees as base classifier).

        Parameters
        -----------------------------

        :param league_id: The created league id.
        :param model_id: The provided model id by the user.
        :param target_type: The classification target type.
        :param calibrate_probabilities: Whether to calibrate output probabilities.
        :param normalizer: The input data normalizer placeholder.
        :param sampler: The sampler method for training data.
        :param n_estimators: int. Number of decision tree classifiers.
        :param criterion: string. The Decision Tree criterion (can be "gini", "log_loss" or "entropy").
        :param min_samples_leaf: int. The minimum number of samples to form a leaf node.
        :param min_samples_split: int. The minimum number of samples in a tree node to create a branch (split).
        :param max_features: str | None. The maximum number of features (None, sqrt, log2).
                             If None, all features are considered, else only the top K features are used.
        :param max_depth: int | None. The maximum tree dept. If None, there is no depth limit.
                                      Small depth values are ideal for interpretable trees.
        :param class_weight: bool. Whether to assign different weights to minority-majority classes.
                                          It can be good in tasks with unbalanced classes.
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
            criterion: str = 'gini',
            min_samples_leaf: int = 1,
            min_samples_split: int = 2,
            max_features: Optional[str] = None,
            max_depth: Optional[int] = None,
            class_weight: bool = True,
            **kwargs
    ):
        if max_depth == 0:
            max_depth = None

        self._n_estimators = n_estimators
        self._criterion = criterion
        self._min_samples_leaf = min_samples_leaf
        self._min_samples_split = min_samples_split
        self._max_features = max_features
        self._max_depth = max_depth
        self._class_weight = class_weight

        super().__init__(
            league_id=league_id,
            model_id=model_id,
            target_type=target_type,
            calibrate_probabilities=calibrate_probabilities,
            normalizer=normalizer,
            sampler=sampler,
            **kwargs
        )

    def build_classifier(self, input_size: int, num_classes: int) -> RandomForestClassifier:
        """ Builds a Random Forest classification model. """

        return RandomForestClassifier(
            n_estimators=self._n_estimators,
            criterion=self._criterion,
            min_samples_leaf=self._min_samples_leaf,
            min_samples_split=self._min_samples_split,
            max_features=self._max_features,
            max_depth=self._max_depth,
            class_weight='balanced' if self._class_weight else None,
            n_jobs=-1,
            random_state=0
        )

    def get_feature_importances(self) -> np.ndarray:
        """ Gets feature importances of Random Forest. """

        if self._calibrate_probabilities:
            return self.classifier.calibrated_classifiers_[0].estimator.feature_importances_
        else:
            return self._classifier.feature_importances_

    def get_estimator(self, estimator_id: int = 0) -> BaseEstimator:
        """ Gets the specified estimator (Decision Tree) of Random Forest. """

        if self._calibrate_probabilities:
            return self._classifier.calibrated_classifiers_[0].estimator.estimators_[estimator_id]
        else:
            return self._classifier.estimators_[estimator_id]

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == 'n_estimators':
            return {'low': 50, 'high': 500, 'step': 50}
        elif param == 'criterion':
            return ['gini', 'entropy', 'log_loss']
        elif param == 'min_samples_leaf':
            return {'low': 1, 'high': 30, 'step': 2}
        elif param == 'min_samples_split':
            return {'low': 2, 'high': 30, 'step': 2}
        elif param == 'max_features':
            return [None, 'sqrt', 'log2']
        elif param == 'max_depth':
            return {'low': 0, 'high': 15, 'step': 1}
        elif param == 'class_weight':
            return [True, False]
        else:
            raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.update({
            'n_estimators': self._n_estimators,
            'criterion': self._criterion,
            'min_samples_leaf': self._min_samples_leaf,
            'min_samples_split': self._min_samples_split,
            'max_features': self._max_features,
            'max_depth': self._max_depth,
            'class_weight': self._class_weight
        })
        return model_config
