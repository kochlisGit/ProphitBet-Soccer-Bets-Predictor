import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from typing import Optional
from matplotlib.pyplot import Axes
from src.models.model import ClassificationModel
from src.interpretability.explainer import ClassifierExplainer


class ExtremeBoostingExplainer(ClassifierExplainer):
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
        scores_df = pd.DataFrame({'Features': self._df.columns.tolist(), 'Impurity': scores})
        scores_df = scores_df.sort_values(by='Impurity', ascending=False, ignore_index=True).iloc[: self._max_features]

        # Generate importance bar plot.
        _, ax = plt.subplots(constrained_layout=True)
        ax = sns.barplot(x=scores_df['Impurity'], y=scores_df['Features'], palette=colormap, ax=ax)
        ax.set_ylabel(None)
        ax.tick_params(axis='both', labelsize='small')
        ax.set_title(f'Impurity Bar Plot for {self._model.target_type.name}')
        return ax
