import numpy as np
import optuna
from typing import Any, Dict, List, Optional, Union
from xgboost import XGBClassifier
from src.core.preprocessing.utils import SamplerType, TargetType
from src.core.models.model import Classifier


class XGBoost(Classifier):
    """ Implementation of the Extreme Boosting classifier.

       Attributes
       -----------
        :param target_type: (TargetType) classification target type.
        :param sampler: (str | SamplerType | BaseSampler) sampler method for training data.
        :param calibrate_probabilities: (bool) whether to calibrate output probabilities.
        :param n_estimators: (int) number of decision tree classifiers.
        :param max_depth: (Optional[int]) The maximum tree dept. If None, there is no depth limit.
                                      Small depth values are ideal for interpretable trees.
        :param min_child_weight: (int) minimum weight of each child estimator.
        :param learning_rate: (float) learning_rate of the gradient descent step, between 0.0 and 1.0.
        :param subsample: (float) fraction of training rows sampled per tree; values less than 1.0 help prevent overfitting.
        :param colsample_bytree: (float) fraction of features sampled for each tree.
        :param reg_lambda: (float) the lambda regularization value.
        :param reg_alpha: (float) the alpha regularization value.
   """

    def __init__(
            self,
            target_type: TargetType,
            sampler: Optional[SamplerType] = None,
            calibrate_probabilities: bool = True,
            n_estimators: int = 100,
            max_depth: int = 0,
            min_child_weight: int = 1,
            learning_rate: float = 0.3,
            subsample: float = 1.0,
            colsample_bytree: float = 1.0,
            reg_lambda: float = 1.0,
            reg_alpha: float = 0.0,
            **kwargs
    ):
        super().__init__(
            target_type=target_type,
            normalizer=None,
            sampler=sampler,
            calibrate_probabilities=calibrate_probabilities,
            n_jobs=-1,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha
        )

    @staticmethod
    def get_suggested_hyperparameters() -> Dict[str, Union[List[Any], Dict[str, Any]]]:
        """ Returns a suggested set of all hyperparameters (Dict[str, values]) for the classifier.  """

        return {
            'sampler': [sample.value for sample in SamplerType],
            'calibrate_probabilities': [True, False],
            'n_estimators': {'low': 50, 'high': 1000, 'step': 50},
            'max_depth': {'low': 1, 'high': 15, 'step': 1},
            'min_child_weight':  {'low': 1, 'high': 20, 'step': 1},
            'learning_rate': {'low': 0.005, 'high': 0.3, 'step': 0.005},
            'subsample': {'low': 0.6, 'high': 1.0, 'step': 0.2},
            'colsample_bytree': {'low': 0.6, 'high': 1.0, 'step': 0.2},
            'reg_lambda': {'low': 0.1, 'high': 10.0, 'step': 0.1},
            'reg_alpha': {'low': 0.0, 'high': 10.0, 'step': 0.1}
        }

    def build(
            self,
            input_size: int,
            num_classes: int,
            seed: int = 0,
            trial: Optional[optuna.Trial] = None
    ) -> XGBClassifier:
        return XGBClassifier(
            booster='gbtree',
            n_estimators=self._model_config['n_estimators'],
            max_depth=self._model_config['max_depth'],
            min_child_weight=self._model_config['min_child_weight'],
            learning_rate=self._model_config['learning_rate'],
            subsample=self._model_config['subsample'],
            colsample_bytree=self._model_config['colsample_bytree'],
            reg_lambda=self._model_config['reg_lambda'],
            reg_alpha=self._model_config['reg_alpha'],
            random_state=seed,
            n_jobs=-1,
            verbosity=0
        )

    def get_importances(self) -> np.ndarray:
        """ Returns the feature importances of the XGBoost model. """

        if self._model_config['calibrate_probabilities']:
            importances = [
                calibrated.estimator.feature_importances_
                for calibrated in self._classifier.calibrated_classifiers_
            ]

            return np.mean(importances, axis=0)
        return self._classifier.feature_importances_
