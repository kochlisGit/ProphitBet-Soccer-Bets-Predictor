import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from typing import Optional
from matplotlib.pyplot import Axes
from shap.maskers import Independent
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.models.model import ClassificationModel
from src.interpretability.explainer import ClassifierExplainer


class DiscriminantAnalysisExplainer(ClassifierExplainer):
    def __init__(self, model: ClassificationModel, df: pd.DataFrame):
        super().__init__(model=model, df=df)

    def _compute_shap_values(self) -> Optional[shap.Explanation]:
        """ Returns shap values of Linear-based explainer (for linear kernel) or kernel explainer (for other kernels). """

        clf = self._model.classifier if not self._model.calibrate_probabilities else self._model.classifier.calibrated_classifiers_[0].estimator
        kernel = self._model.decision_boundary

        if kernel == 'linear':
            explainer = shap.LinearExplainer(model=clf, masker=Independent(data=self._df))
        else:
            return None

        return explainer(self._df)

    def visualize_model(self) -> Axes:
        n_components = min(self._num_classes - 1, 2)
        clf = LinearDiscriminantAnalysis(solver='svd', n_components=n_components, store_covariance=True)
        clf.fit(self._x, self._y)

        if clf is None:
            pass

        x_transformed = clf.transform(self._x)

        if x_transformed.shape[0] == 1:
            zeros = np.zeros_like(x_transformed, dtype=x_transformed.dtype)
            x_transformed = np.hstack([x_transformed, zeros])

        df_2d = pd.DataFrame(data=x_transformed, columns=['x', 'y'])
        df_2d['class'] = self._y
        _, ax = plt.subplots(constrained_layout=True)
        ax = sns.scatterplot(data=df_2d, x='x', y='y', hue='class', palette='coolwarm', ax=ax)

        colormap = plt.cm.coolwarm
        num_classes = self._num_classes
        colours = [colormap(i/(num_classes - 1)) for i in range(num_classes)]
        legend_handles = [mpatches.Patch(color=colours[i], label=name) for i, name in enumerate(self._class_names)]
        ax.legend(handles=legend_handles, title='Class', loc='upper right')
        ax.set_title(f'Visualization of LDA Classifier in 2 Dimensions.')
        return ax
