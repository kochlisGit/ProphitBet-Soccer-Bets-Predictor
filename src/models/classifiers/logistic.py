import numpy as np
from typing import Any, Dict, List, Optional, Union
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from src.preprocessing.utils.sampling import SamplerType
from src.preprocessing.utils.target import TargetType
from src.models.model import ClassificationModel


class LogisticRegressor(ClassificationModel):
    """ Implementation of Logistic Regression Classifier.

        Parameters
        -----------------------------

        :param league_id: The created league id.
        :param model_id: The provided model id by the user.
        :param target_type: The classification target type.
        :param calibrate_probabilities: Whether to calibrate output probabilities.
        :param normalizer: The input data normalizer placeholder.
        :param sampler: The sampler method for training data.
        :param penalty: str | None. The penalty method. It can either be "l1", "l2", "elastic" (l1 + l2) or None.
                                    l1: minimizes (WX - y)^2 - |W|. l2 minimizes (WX - y)^2 - W^2. elastic uses both
                                    l1 and l2. None does not use a regularization term (minimizes (WX - Y)^2).
    """

    def __init__(
            self,
            league_id: str,
            model_id: str,
            target_type: TargetType,
            calibrate_probabilities: bool,
            normalizer: Optional[TransformerMixin] = None,
            sampler: Optional[SamplerType] = None,
            penalty: Optional[str] = None,
            **kwargs
    ):
        self._penalty = penalty
        self._cs = [1e-4, 1e-2, 1e-1, 1.0, 10]

        super().__init__(
            league_id=league_id,
            model_id=model_id,
            target_type=target_type,
            calibrate_probabilities=calibrate_probabilities,
            normalizer=normalizer,
            sampler=sampler,
            **kwargs
        )

    def build_classifier(self, input_size: int, num_classes: int) -> LogisticRegression:
        """ Builds a Logistic Regression model. """

        if self._penalty is None:
            return LogisticRegression(penalty=None, max_iter=5000, n_jobs=-1)
        elif self._penalty == 'l2' or self._penalty == 'elastic':
            return LogisticRegressionCV(Cs=self._cs, penalty=self._penalty, solver='lbfgs', max_iter=5000, random_state=0, n_jobs=-1)
        elif self._penalty == 'l1':
            return LogisticRegressionCV(Cs=self._cs, penalty=self._penalty, solver='saga', max_iter=5000, random_state=0, n_jobs=-1)
        else:
            raise ValueError(f'Undefined penalty type: "{self._penalty}".')

    def get_coefficients(self) -> np.ndarray:
        """ Returns the coefficients of Logistic Regression model. """

        if self._calibrate_probabilities:
            coeffs = self._classifier.calibrated_classifiers_[0].estimator.coef_
        else:
            coeffs = self._classifier.coef_

        return np.abs(coeffs)

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == 'penalty':
            return [None, 'l1', 'l2']
        else:
            raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.update({
            'penalty': self._penalty,
        })
        return model_config
