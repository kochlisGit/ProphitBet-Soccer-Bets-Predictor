from sklearn.tree import DecisionTreeClassifier, plot_tree
from models.model import ScikitModel


class DecisionTree(ScikitModel):
    def __init__(
            self,
            model_id: str,
            criterion: str = 'gini',
            min_samples_leaf: int = 1,
            min_samples_split: int = 2,
            max_features: str or None = None,
            max_depth: int or None = None,
            class_weight: str or None = None,
            calibrate_probabilities: bool = False,
            **kwargs
    ):
        assert criterion == 'gini' or criterion == 'entropy' or criterion == 'log_loss', \
            f'Not supported criterion: "{criterion}"'
        assert class_weight is None or class_weight == 'balanced' or class_weight == 'None',\
            f'Not supported class weight: "{class_weight}"'
        assert min_samples_leaf > 0, f'min_samples_leaf should be positive, got {min_samples_leaf}'
        assert min_samples_split > 0, f'min_samples_split should be positive, got {min_samples_split}'
        assert max_features is None or max_features == 'sqrt' or max_features == 'log2' or max_features == 'None', \
            f'max_features is expected to be None or sqrt/log2, got {max_features}'
        assert max_depth is None or max_depth >= 0, f'max_depth is expected to be None or positive integer, got {max_depth}'

        self._criterion = criterion
        self._min_samples_leaf = min_samples_leaf
        self._min_samples_split = min_samples_split
        self._max_features = None if max_features == 'None' else max_features
        self._max_depth = None if max_depth == 0 else max_depth
        self._class_weight = None if class_weight == 'None' else class_weight

        super().__init__(
            model_id=model_id,
            model_name='decision-tree',
            calibrate_probabilities=calibrate_probabilities,
            **kwargs
        )

    def _build_estimator(self, input_size: int, num_classes: int) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            criterion=self._criterion,
            min_samples_leaf=self._min_samples_leaf,
            min_samples_split=self._min_samples_split,
            max_features=self._max_features,
            max_depth=self._max_depth
        )

    def plot_tree(self, ax):
        plot_tree(decision_tree=self._model, ax=ax)
