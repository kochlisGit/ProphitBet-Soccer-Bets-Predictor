import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from typing import List, Optional
from matplotlib.pyplot import Axes
from shap.maskers import Independent
from sklearn.inspection import DecisionBoundaryDisplay
from src.models.model import ClassificationModel
from src.interpretability.explainer import ClassifierExplainer


class SVMExplainer(ClassifierExplainer):
    def __init__(self, model: ClassificationModel, df: pd.DataFrame):
        super().__init__(model=model, df=df)

    def _compute_shap_values(self) -> Optional[shap.Explanation]:
        """ Returns shap values of Linear-based explainer (for linear kernel) or kernel explainer (for other kernels). """

        clf = self._model.classifier if not self._model.calibrate_probabilities else self._model.classifier.calibrated_classifiers_[0].estimator
        kernel = self._model.kernel

        if kernel == 'linear':
            explainer = shap.LinearExplainer(model=clf, masker=Independent(data=self._df))
        else:
            return None

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

    def visualize_model(self, features: List[str]) -> Axes:
        """ Generates the SVM plot along with the decision boundaries and support vectors. """

        if len(features) != 2:
            raise ValueError(f'Features is expected to be a list of 2 column names, got {features}')

        # Selecting non-nan (input, target) data.
        columns = self._df.columns
        x = self._x[:, [columns.get_loc(features[0]), columns.get_loc(features[1])]]
        y = self._y

        # Fitting the classifier into the data.
        num_classes = self._num_classes
        clf = self._model.build_classifier(input_size=2, num_classes=num_classes)
        clf.fit(x, y)

        # Plot decision boundaries.
        colormap = plt.cm.coolwarm
        _, ax = plt.subplots(constrained_layout=True)
        ax = DecisionBoundaryDisplay.from_estimator(
            clf,
            x,
            grid_resolution=200,
            response_method='predict',
            plot_method='pcolormesh',
            cmap=colormap,
            alpha=0.8,
            xlabel=features[0],
            ylabel=features[1],
            ax=ax
        ).ax_

        DecisionBoundaryDisplay.from_estimator(
            clf,
            x,
            class_of_interest=None if num_classes < 3 else 0,
            response_method="decision_function",
            plot_method="contour",
            levels=[-1, 0, 1],
            colors=['k', 'k', 'k'],
            linestyles=['--', '-', '--'],
            linewidths=[2, 2, 2],
            ax=ax
        )
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap=colormap, s=20, edgecolors='k')

        # Plot support vectors.
        ax.scatter(x=clf.support_vectors_[:, 0], y=clf.support_vectors_[:, 1], s=150, facecolors='none', edgecolors="k")

        # Create legend for each label.
        colours = [colormap(i/(num_classes - 1)) for i in range(num_classes)]
        legend_handles = [mpatches.Patch(color=colours[i], label=name) for i, name in enumerate(self._class_names)]

        ax.legend(handles=legend_handles, title='Class', loc='upper right')
        ax.set_title(f'SVM Decision Boundaries with {self._model.kernel} kernel')
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
