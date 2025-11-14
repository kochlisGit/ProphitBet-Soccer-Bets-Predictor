import numpy as np
from typing import Any, Dict, List, Optional, Union
from sklearn.base import TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from src.preprocessing.utils.sampling import SamplerType
from src.preprocessing.utils.target import TargetType
from src.models.model import ClassificationModel


class DecisionTree(ClassificationModel):
    """ Implementation of Decision Tree Classifier.

        Parameters
        -----------------------------

        :param league_id: The created league id.
        :param model_id: The provided model id by the user.
        :param target_type: The classification target type.
        :param calibrate_probabilities: Whether to calibrate output probabilities.
        :param normalizer: The input data normalizer placeholder.
        :param sampler: The sampler method for training data.
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

    def build_classifier(self, input_size: int, num_classes: int) -> DecisionTreeClassifier:
        """ Builds a Decision Tree model. """

        return DecisionTreeClassifier(
            criterion=self._criterion,
            min_samples_leaf=self._min_samples_leaf,
            min_samples_split=self._min_samples_split,
            max_features=self._max_features,
            max_depth=self._max_depth,
            class_weight='balanced' if self._class_weight else None,
            random_state=0
        )

    def get_feature_importances(self) -> np.ndarray:
        """ Gets feature importances of Decision Tree. """

        if self._calibrate_probabilities:
            return self._classifier.calibrated_classifiers_[0].estimator.feature_importances_
        else:
            return self._classifier.feature_importances_

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == 'criterion':
            return ['gini', 'entropy', 'log_loss']
        elif param == 'min_samples_leaf':
            return {'low': 1, 'high': 35, 'step': 2}
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
            'criterion': self._criterion,
            'min_samples_leaf': self._min_samples_leaf,
            'min_samples_split': self._min_samples_split,
            'max_features': self._max_features,
            'max_depth': self._max_depth,
            'class_weight': self._class_weight
        })
        return model_config
