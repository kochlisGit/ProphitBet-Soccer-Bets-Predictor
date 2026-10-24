import numpy as np
from typing import Any, Dict, List, Optional, Union
from sklearn.tree import DecisionTreeClassifier
from src.core.preprocessing.utils import SamplerType, TargetType
from src.core.models.model import Classifier


class DecisionTree(Classifier):
    """ Implementation of the Decision Tree classifier.

       Attributes
       -----------
        :param target_type: (TargetType) classification target type.
        :param sampler: (str | SamplerType | BaseSampler) sampler method for training data.
        :param calibrate_probabilities: (bool) whether to calibrate output probabilities.
        :param criterion: (str) optimization criterion (can be "gini", "log_loss" or "entropy").
        :param min_samples_leaf: (int) minimum number of samples to form a leaf node.
        :param min_samples_split: (int) minimum number of samples in a tree node to create a branch (split).
        :param max_features: (Optional[int]) maximum number of features (None, sqrt, log2).
                             If None, all features are considered, else only the top K features are used.
        :param max_depth: (Optional[int]) The maximum tree dept. If None, there is no depth limit.
                                      Small depth values are ideal for interpretable trees.
        :param class_weight: (bool) whether to assign different weights to minority-majority classes.
                                          It can be good in tasks with unbalanced classes.
   """

    def __init__(
            self,
            target_type: TargetType,
            sampler: Optional[SamplerType] = None,
            calibrate_probabilities: bool = True,
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

        super().__init__(
            target_type=target_type,
            normalizer=None,
            sampler=sampler,
            calibrate_probabilities=calibrate_probabilities,
            n_jobs=None,
            criterion=criterion,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            max_depth=max_depth,
            class_weight=class_weight
        )

    @staticmethod
    def get_suggested_hyperparameters() -> Dict[str, Union[List[Any], Dict[str, Any]]]:
        """ Returns a suggested set of all hyperparameters (Dict[str, values]) for the classifier.  """

        return {
            'sampler': [sample.value for sample in SamplerType],
            'calibrate_probabilities': [True, False],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'min_samples_leaf': {'low': 1, 'high': 35, 'step': 2},
            'min_samples_split': {'low': 2, 'high': 30, 'step': 2},
            'max_features': [None, 'sqrt', 'log2'],
            'max_depth': {'low': 0, 'high': 10, 'step': 1},
            'class_weight': [True, False]
        }

    def build(self, input_size: int, num_classes: int, seed: int = 0) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            criterion=self._model_config['criterion'],
            min_samples_leaf=self._model_config['min_samples_leaf'],
            min_samples_split=self._model_config['min_samples_split'],
            max_features=self._model_config['max_features'],
            max_depth=self._model_config['max_depth'],
            class_weight='balanced' if self._model_config['class_weight'] else None,
            random_state=seed
        )

    def get_importances(self) -> np.ndarray:
        """ Returns the feature importances of the tree model. """

        if self._model_config['calibrate_probabilities']:
            importances = [
                calibrated.estimator.feature_importances_
                for calibrated in self._classifier.calibrated_classifiers_
            ]

            return np.mean(importances, axis=0)
        return self._classifier.feature_importances_
