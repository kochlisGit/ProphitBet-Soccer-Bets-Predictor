from sklearn.svm import SVC
from models.model import ScikitModel


class SupportVectorMachine(ScikitModel):
    def __init__(
            self,
            model_id: str,
            C: float = 1.0,
            gamma: str or float = 'scale',
            kernel: str = 'rbf',
            class_weight: str or None = None,
            calibrate_probabilities: bool = False,
            **kwargs
    ):
        assert C > 0, f'C is expected to be positive, got {C}'
        assert gamma == 'scale' or gamma == 'auto' or (isinstance(gamma, float) and gamma > 0), \
            f'gamma should equal "scale" or "auto" or should be a float value, got {gamma}'
        assert kernel == 'linear' or kernel == 'poly' or kernel == 'rbf' or kernel == 'sigmoid', \
            f'Not supported kernel function: "{kernel}"'
        assert class_weight is None or class_weight == 'balanced' or class_weight == 'None', \
            f'Not supported class weight: "{class_weight}"'

        self._C = C
        self._gamma = gamma
        self._kernel = kernel
        self._class_weight = None if class_weight == 'None' else class_weight

        super().__init__(
            model_id=model_id,
            model_name='support-vector-machine',
            calibrate_probabilities=calibrate_probabilities,
            **kwargs
        )

    def _build_estimator(self, input_size: int, num_classes: int) -> SVC:
        return SVC(
            C=self._C,
            gamma=self._gamma,
            kernel=self._kernel,
            class_weight=self._class_weight,
            probability=True,
            random_state=0
        )

    def get_model_coefficients(self):
        return self._model.coef_
