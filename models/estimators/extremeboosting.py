import numpy as np
from xgboost import XGBClassifier
from models.model import ScikitModel


class XGBoost(ScikitModel):
    def __init__(
            self,
            model_id: str,
            n_estimators: int = 100,
            learning_rate: float = 0.3,
            max_depth: int = 6,
            min_child_weight: int = 1,
            lambda_regularization: float = 1,
            alpha_regularization: float = 0,
            calibrate_probabilities: bool = False,
            **kwargs
    ):
        assert 0 < learning_rate < 1, f'learning_rate is expected to be between 0 and 1, got {learning_rate}'
        assert 0 < max_depth, f'max_depth is expected to be positive, got {max_depth}'
        assert 0 < min_child_weight, f'min_child_weight is expected to be larger than 1, got {min_child_weight}'
        assert 0 <= lambda_regularization, f'lambda_reg is expected to be positive, got {lambda_regularization}'
        assert 0 <= alpha_regularization, f'alpha_reg is expected to be positive, got {alpha_regularization}'

        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._min_child_weight = min_child_weight
        self._lambda_regularization = lambda_regularization
        self._alpha_regularization = alpha_regularization

        super().__init__(
            model_id=model_id,
            model_name='xgboost',
            calibrate_probabilities=calibrate_probabilities,
            **kwargs
        )

    def _build_estimator(self, input_size: int, num_classes: int) -> XGBClassifier:
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

    def get_feature_importance_scores(self) -> np.ndarray:
        return self._model.feature_importances_
