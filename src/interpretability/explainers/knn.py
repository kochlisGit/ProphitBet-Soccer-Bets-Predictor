import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from typing import List, Optional
from matplotlib.pyplot import Axes
from sklearn.metrics import pairwise_distances
from src.models.model import ClassificationModel
from src.interpretability.explainer import ClassifierExplainer


class KNNExplainer(ClassifierExplainer):
    def __init__(self, model: ClassificationModel, df: pd.DataFrame):
        super().__init__(model=model, df=df)

    def _compute_shap_values(self) -> Optional[shap.Explanation]:
        """ Returns shap values of kernel explainer. """

        return None

    def visualize_model(self, features: List[str], match_index: int) -> Axes:
        """ Visualizes KNN model when applied to 2 features. Requires validation in case match_index contains nans. """

        if len(features) != 2:
            raise ValueError(f'Features is expected to be a list of 2 column names, got {features}')

        # Selecting non-nan (input, target) data.
        columns = self._df.columns
        x = self._x[:, [columns.get_loc(features[0]), columns.get_loc(features[1])]]
        y = self._y

        # Computing distances.
        metric = 'euclidean' if self._model.p == 2 else 'manhattan'
        distances = pairwise_distances(
            x[match_index].reshape(1, -1),
            x,
            metric=metric,
            n_jobs=-1
        ).ravel()

        # Computing nearest neighbors based on their distances (the self-distance is excluded).
        distances[match_index] = np.inf
        n = self._model.n_neighbors
        nn_indices = np.argpartition(distances, n)[:n]

        # Fetching nearest neighbors.
        neighbor_distances = distances[nn_indices]
        neighbor_targets = y[nn_indices]

        # Voting the selected match class (using uniform or weighted method).
        if self._model.weights == 'uniform':
            counts = np.bincount(neighbor_targets)
            y_pred = counts.argmax()
        else:
            weights = 1.0/(neighbor_distances + 1e-12)
            scores = {}
            for cls, w in zip(neighbor_targets, weights):
                scores[cls] = scores.get(cls, 0.0) + w
            y_pred = max(scores.items(), key=lambda kv: (kv[1], -kv[0]))[0]

        # Visualize KNN model.
        match_y_true = y[match_index]
        y[match_index] = y_pred
        mask = np.ones_like(y, dtype=bool)
        mask[nn_indices] = False
        mask[match_index] = False

        # Visualize all matches.
        _, ax = plt.subplots(constrained_layout=True)
        ax = sns.scatterplot(x=x[mask, 0], y=x[mask, 1], hue=y[mask], palette='coolwarm', s=20, ax=ax)

        # Visualize all K-Neighbors.
        ax.scatter(x[nn_indices, 0], x[nn_indices, 1], c=y[nn_indices], cmap='coolwarm', s=30)

        # Visualize selected match.
        ax.scatter(x[match_index, 0], x[match_index, 1], c=y[match_index], cmap='coolwarm', edgecolor='k', s=80)

        # Visualize distance (if manhattan (p == 1) then draw rectangle, else (p == 2) draw circle for euclidean).
        r = neighbor_distances[0]

        if metric == 'manhattan':
            rect = plt.Rectangle((x[match_index, 0] - r, x[match_index, 1] - r), 2*r, 2*r, fill=False, linestyle='--', color='k')
            ax.add_patch(rect)
        else:
            circle = plt.Circle((x[match_index, 0], x[match_index, 1]), r, fill=False, linestyle='--', color='k')
            ax.add_patch(circle)

        y[match_index] = match_y_true

        ax.legend()
        ax.grid()
        ax.set_title(f'KNN Decision Boundaries with {metric} metric')
        return ax
