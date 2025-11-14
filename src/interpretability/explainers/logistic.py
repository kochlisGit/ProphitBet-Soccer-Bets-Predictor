import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from typing import Optional
from matplotlib.pyplot import Axes
from shap.maskers import Independent
from src.models.model import ClassificationModel
from src.interpretability.explainer import ClassifierExplainer


class LogisticRegressionExplainer(ClassifierExplainer):
    def __init__(self, model: ClassificationModel, df: pd.DataFrame):
        super().__init__(model=model, df=df)

    def _compute_shap_values(self) -> shap.Explanation:
        """ Returns shap values of Linear-based explainer. """

        clf = self._model.classifier if not self._model.calibrate_probabilities else self._model.classifier.calibrated_classifiers_[0].estimator
        explainer = shap.LinearExplainer(model=clf, masker=Independent(data=self._df))
        return explainer(self._df)

    def coefficients_bar_plot(self, colormap: Optional[str] = None) -> Axes:
        """ Generates the coefficient bar plot. """

        coef = self._model.get_coefficients()
        classes = self._class_names
        feature_names = self._df.columns.tolist()
        n_classes = self._num_classes

        # Generate bar plots for binary classification tasks.
        if n_classes < 3:
            coef_df = pd.DataFrame({'Features': feature_names, 'Coefficients': coef[0]})
            ax = self._generate_coefficient_plot(coef_df=coef_df, vertical_labels=True, ax=None, colormap=colormap)
            ax.set_title(f'Top-{self._max_features} Feature Importance Bar Plot for {classes[0]}-{classes[1]}')
        else:
            fig, axes = plt.subplots(nrows=n_classes, ncols=1)
            fig.suptitle(f'Top-{self._max_features} Feature Importance Bar Plot for {classes}')
            for idx, cls_label in enumerate(classes):
                ax = axes[idx]
                coef_df = pd.DataFrame({'Features': feature_names, 'Coefficients': coef[idx]})
                _ = self._generate_coefficient_plot(coef_df=coef_df, vertical_labels=False, ax=ax, colormap=colormap)
                ax.set_ylabel(f'{cls_label} Coefficients')
            ax = axes
        return ax

    def visualize_model(self, feature: str) -> Axes:
        """ Generates the logistic regression plot along with the linear model and the predicted probabilities. """

        # Extract non-nan (input, target) pairs.
        x = np.expand_dims(self._x[:, self._df.columns.get_loc(feature)], axis=-1)
        y = self._y

        # Train logistic regression classifier.
        num_classes = len(self._class_names)
        clf = self._model.build_classifier(input_size=2, num_classes=num_classes)
        clf.fit(x, y)

        y_prob = clf.predict_proba(x)
        plot_df = pd.DataFrame({
            'x': x[:, 0],
            **{
                class_name: y_prob[:, i] for i, class_name in enumerate(self._class_names)
            }
        })

        # Plotting probabilities (logistic regression).
        fig, ax = plt.subplots(constrained_layout=True)
        for class_name in self._class_names:
            ax = sns.scatterplot(data=plot_df, x='x', y=class_name, s=20, edgecolor='k', label=class_name, ax=ax)

        ax.set_xlabel(feature)
        ax.set_ylabel('Probability')
        ax.set_ylim(0.0, 1.0)
        ax.grid()
        ax.legend(frameon=True)
        ax.set_title(f'Logistic Regression Probability Estimations')
        return ax

    def _generate_coefficient_plot(
            self,
            coef_df: pd.DataFrame,
            vertical_labels: bool = False,
            ax: Optional[Axes] = None,
            colormap: Optional[str] = None
    ) -> Axes:
        """ Generates the coefficient bar plot of the Top-K features. """

        if vertical_labels:
            x_label = 'Coefficients'
            y_label = 'Features'
        else:
            x_label = 'Features'
            y_label = 'Coefficients'

        coef_df = coef_df.sort_values(by='Coefficients', ascending=False, ignore_index=True).iloc[: self._max_features]
        ax = sns.barplot(
            x=coef_df[x_label],
            y=coef_df[y_label],
            ax=ax,
            palette=colormap
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.tick_params(axis='both', labelsize='small')
        return ax
