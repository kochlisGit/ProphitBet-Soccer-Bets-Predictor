import numpy as np
from typing import Any, Dict, List, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from src.preprocessing.utils.sampling import SamplerType
from src.preprocessing.utils.target import TargetType
from src.models.model import ClassificationModel


class SVM(ClassificationModel):
    """ Implementation of Support Vector Machines (SVM) Classifier.
    
        Parameters
        -----------------------------
        
        :param league_id: The created league id.
        :param model_id: The provided model id by the user.
        :param target_type: The classification target type.
        :param calibrate_probabilities: Whether to calibrate output probabilities.
        :param normalizer: The input data normalizer placeholder.
        :param sampler: The sampler method for training data.
        :param kernel: str | None. The kernel function of SVM. Supported kernels are: None, "linear", "rbf", "poly" or 
                                   "sigmoid".
        :param degree: int. The polynomial degree of kernel function (if kernel == "poly").
        :param gamma: int. The gamma regula
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
            kernel: str = 'linear',
            degree: int = 3,
            gamma: float = 1.0,
            class_weight: bool = True,
            **kwargs
    ):
        self._kernel = kernel
        self._degree = degree
        self._gamma = gamma
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

    @property
    def kernel(self) -> str:
        return self._kernel

    def build_classifier(self, input_size: int, num_classes: int) -> BaseEstimator:
        """ Builds an SVM classification model. """

        return SVC(
            C=self._gamma,
            degree=self._degree,
            kernel=self._kernel,
            class_weight='balanced' if self._class_weight else None,
            probability=True,
            random_state=0
        )

    def get_coefficients(self) -> np.ndarray:
        """ Returns the coefficients of SVM model. """

        if self._calibrate_probabilities:
            coeffs = self._classifier.calibrated_classifiers_[0].estimator.coef_
        else:
            coeffs = self._classifier.coef_

        return np.abs(coeffs)

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == 'kernel':
            return ['linear', 'rbf', 'poly', 'sigmoid']
        elif param == 'degree':
            return {'low': 3, 'high': 6, 'step': 1}
        elif param == 'gamma':
            return {'low': 0.1, 'high': 2.0, 'step': 0.1}
        elif param == 'class_weight':
            return [True, False]
        else:
            raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.update({
            'kernel': self._kernel,
            'degree': self._degree,
            'gamma': self._gamma,
            'class_weight': self._class_weight
        })
        return model_config
