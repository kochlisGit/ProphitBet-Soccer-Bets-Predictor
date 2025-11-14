import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from typing import Optional
from matplotlib.pyplot import Axes
from sklearn.base import BaseEstimator
from sklearn.tree import plot_tree
from src.models.model import ClassificationModel
from src.interpretability.explainer import ClassifierExplainer


class DecisionTreeExplainer(ClassifierExplainer):
    def __init__(self, model: ClassificationModel, df: pd.DataFrame):
        super().__init__(model=model, df=df)

    def _compute_shap_values(self) -> shap.Explanation:
        """ Returns shap values of Tree-based explainer. """

        clf = self._model.classifier if not self._model.calibrate_probabilities else self._model.classifier.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(model=clf)
        return explainer(self._df)

    def feature_impurity_bar_plot(self, colormap: Optional[str] = None) -> Axes:
        """ Generates an impurity-based bar plot that highlights the importance of each feature. """

        scores = self._model.get_feature_importances()
        ax = self._plot_impurity_scores(scores=scores, colormap=colormap)
        return ax

    def plot_tree_rules(self, max_depth: int = 3) -> Axes:
        """ Plot the extracted rules by tree. """

        clf = self._model.classifier if not self._model.calibrate_probabilities else self._model.classifier.calibrated_classifiers_[0].estimator
        ax = self._plot_tree(estimator=clf, max_depth=max_depth)
        return ax

    def _plot_impurity_scores(self, scores: Optional[np.ndarray] = None, colormap: Optional[str] = None) -> Axes:
        """ Generates an impurity-based bar plot that highlights the importance of each feature. """

        scores_df = pd.DataFrame({'Features': self._df.columns.tolist(), 'Impurity': scores})
        scores_df = scores_df.sort_values(by='Impurity', ascending=False, ignore_index=True).iloc[: self._max_features]

        # Generate importance bar plot.

        ax = sns.barplot(x=scores_df['Impurity'], y=scores_df['Features'], palette=colormap)
        ax.set_ylabel(None)
        ax.tick_params(axis='both', labelsize='small')
        ax.set_title(f'Impurity Bar Plot for {self._model.target_type.name}')
        return ax

    def _plot_tree(self, estimator: BaseEstimator, max_depth: int) -> Axes:
        """ Plot the extracted rules by tree. """

        _, ax = plt.subplots(constrained_layout=True)
        ax = plot_tree(
            decision_tree=estimator,
            max_depth=max_depth,
            feature_names=self._df.columns.tolist(),
            class_names=self._class_names,
            proportion=True,
            rounded=True,
            fontsize=6,
            ax=ax
        )
        return ax
