from sklearn.linear_model import LogisticRegressionCV
from models.model import ScikitModel


class LogisticRegression(ScikitModel):
    def __init__(
            self,
            model_id: str,
            penalty: str = 'l2',
            class_weight: str or None = None,
            calibrate_probabilities: bool = False,
            **kwargs
    ):
        assert class_weight is None or class_weight == 'balanced' or class_weight == 'None', \
            f'Not supported class weight: "{class_weight}"'

        if penalty == 'l1':
            self._solver = 'liblinear'
        else:
            assert penalty == 'l2', f'Logistic Regression CV supports either l1 or l2, got "{penalty}"'
            self._solver = 'lbfgs'

        self._class_weight = None if class_weight == 'None' else class_weight
        self._penalty = penalty

        super().__init__(
            model_id=model_id,
            model_name='logistic-regression',
            calibrate_probabilities=calibrate_probabilities,
            **kwargs
        )

    def _build_estimator(self, input_size: int, num_classes: int) -> LogisticRegressionCV:
        return LogisticRegressionCV(
            penalty=self._penalty,
            solver=self._solver,
            n_jobs=-1,
            random_state=0,
            class_weight=self._class_weight
        )

    def get_model_coefficients(self):
        return self._model.coef_
