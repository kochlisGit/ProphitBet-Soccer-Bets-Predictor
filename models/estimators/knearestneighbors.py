from sklearn.neighbors import KNeighborsClassifier
from models.model import ScikitModel


class KNearestNeighbors(ScikitModel):
    def __init__(
            self,
            model_id: str,
            n_neighbors: int = 5,
            weights: str = 'uniform',
            calibrate_probabilities: bool = False,
            **kwargs
    ):
        assert n_neighbors > 0 and n_neighbors % 2 != 0, f'"k" neighbors is expected to be an odd number, got {n_neighbors}'
        assert weights == 'uniform' or weights == 'distance', f'Not supported weights: "{weights}"'

        self._n_neighbors = n_neighbors
        self._weights = weights

        super().__init__(
            model_id=model_id,
            model_name='k-nearest-neighbors',
            calibrate_probabilities=calibrate_probabilities,
            **kwargs
        )

    def _build_estimator(self, input_size: int, num_classes: int) -> KNeighborsClassifier:
        return KNeighborsClassifier(
            n_neighbors=self._n_neighbors,
            weights=self._weights,
            n_jobs=-1
        )
