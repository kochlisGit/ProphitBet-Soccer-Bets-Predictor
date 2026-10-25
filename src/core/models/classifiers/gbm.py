import numpy as np
from typing import Any, Dict, List, Optional, Union
from lightgbm import LGBMClassifier
from src.core.preprocessing.utils import SamplerType, TargetType
from src.core.models.model import Classifier


class LightGBM(Classifier):
    """ Implementation of the LightGBM classifier.

        Attributes
        ----------
        :param target_type: (TargetType) classification target type.
        :param sampler: (SamplerType | None) sampler method for the training data.
        :param calibrate_probabilities: (bool) whether to calibrate the output probabilities.
        :param n_estimators: (int) number of boosting iterations.
        :param num_leaves: (int) maximum number of leaves in each tree.
        :param max_depth: (int | None) maximum tree depth. Values <= 0 indicate no depth limit.
        :param min_child_samples: (int) minimum number of samples required in a leaf.
        :param min_child_weight: (float) minimum sum of instance Hessian weights required in a leaf.
        :param learning_rate: (float) boosting learning rate controlling the contribution of each tree.
        :param subsample: (float) fraction of training samples used when bagging is enabled.
        :param subsample_freq: (int) frequency of row subsampling. Values <= 0 disable bagging.
        :param colsample_bytree: (float) fraction of features sampled when constructing each tree.
        :param reg_lambda: (float) L2 regularization coefficient.
        :param reg_alpha: (float) L1 regularization coefficient.
    """

    def __init__(
            self,
            target_type: TargetType,
            sampler: Optional[SamplerType] = None,
            calibrate_probabilities: bool = True,
            n_estimators: int = 100,
            num_leaves: int = 31,
            max_depth: Optional[int] = -1,
            min_child_samples: int = 20,
            min_child_weight: float = 1e-3,
            learning_rate: float = 0.1,
            subsample: float = 1.0,
            subsample_freq: int = 0,
            colsample_bytree: float = 1.0,
            reg_lambda: float = 0.0,
            reg_alpha: float = 0.0,
            **kwargs
    ):
        if max_depth == 0:
            max_depth = -1

        super().__init__(
            target_type=target_type,
            normalizer=None,
            sampler=sampler,
            calibrate_probabilities=calibrate_probabilities,
            n_jobs=-1,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            min_child_weight=min_child_weight,
            learning_rate=learning_rate,
            subsample=subsample,
            subsample_freq=subsample_freq,
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
            'num_leaves': {'low': 7, 'high': 127, 'step': 8},
            'max_depth': {'low': 0, 'high': 15, 'step': 1},
            'min_child_samples': {'low': 5, 'high': 100, 'step': 5},
            'min_child_weight':  {'low': 1, 'high': 20, 'step': 1},
            'learning_rate': {'low': 0.005, 'high': 0.5, 'step': 0.005},
            'subsample': {'low': 0.6, 'high': 1.0, 'step': 0.2},
            'subsample_freq': {'low': 0, 'high': 5, 'step': 1},
            'colsample_bytree': {'low': 0.6, 'high': 1.0, 'step': 0.2},
            'reg_lambda': {'low': 0.1, 'high': 10.0, 'step': 0.1},
            'reg_alpha': {'low': 0.0, 'high': 10.0, 'step': 0.1}
        }

    def build(
            self,
            input_size: int,
            num_classes: int,
            seed: int = 0,
            trial=None
    ) -> LGBMClassifier:
        return LGBMClassifier(
            boosting_type='gbdt',
            n_estimators=self._model_config['n_estimators'],
            num_leaves=self._model_config['num_leaves'],
            max_depth=self._model_config['max_depth'],
            min_child_samples=self._model_config['min_child_samples'],
            min_child_weight=self._model_config['min_child_weight'],
            learning_rate=self._model_config['learning_rate'],
            subsample=self._model_config['subsample'],
            subsample_freq=self._model_config['subsample_freq'],
            colsample_bytree=self._model_config['colsample_bytree'],
            reg_lambda=self._model_config['reg_lambda'],
            reg_alpha=self._model_config['reg_alpha'],
            random_state=seed,
            n_jobs=-1,
            verbosity=-1
        )

    def get_importances(self) -> np.ndarray:
        """ Returns the feature importances of the LightGBM model. """

        if self._model_config['calibrate_probabilities']:
            importances = [
                calibrated.estimator.feature_importances_
                for calibrated in self._classifier.calibrated_classifiers_
            ]

            return np.mean(importances, axis=0)
        return self._classifier.feature_importances_
