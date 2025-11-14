import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from abc import ABC, abstractmethod
from typing import List, Optional
from matplotlib.pyplot import Axes
from sklearn.inspection import DecisionBoundaryDisplay
from src.preprocessing.dataset import DatasetPreprocessor
from src.preprocessing.utils.target import TargetType
from src.models.model import ClassificationModel


class ClassifierExplainer(ABC):
    """ Basic explainer class for classification models. """

    def __init__(self, model: ClassificationModel, df: pd.DataFrame):
        """
            :param model: Trained classifier.
            :param df: Train dataset.
        """

        self._model = model

        if self._model.classifier is None:
            raise ValueError('Expected a pre-trained classifier, got None.')

        target_type = model.target_type

        dataset_preprocessor = DatasetPreprocessor()
        self._x, self._y, _ = dataset_preprocessor.preprocess_dataset(
            df=df,
            target_type=target_type,
            normalizer=self._model.normalizer
        )
        features = df.columns.drop(dataset_preprocessor.non_trainable_columns, errors='ignore')
        self._df = pd.DataFrame(data=self._x, columns=features)

        if target_type == TargetType.RESULT:
            self._class_names = ['H', 'D', 'A']
        elif target_type == TargetType.OVER_UNDER:
            self._class_names = ['U', 'O']
        else:
            raise ValueError(f'Undefined target_type: "{target_type.name}"')

        self._num_classes = len(self._class_names)
        self._shap_values = None
        self._cluster = None
        self._max_features = 15

    @property
    def shap_values(self) -> Optional[np.ndarray]:
        return self._shap_values

    @abstractmethod
    def _compute_shap_values(self) -> shap.Explanation:
        """ Compute the shap values, based on a selected Explainer. """

        pass

    def compute_shap_values(self):
        """ Compute the shap values, based on a selected Explainer. """

        self._shap_values = self._compute_shap_values()

    def boundary_plot(self, features: List[str]) -> Axes:
        """ Generates a boundary plot for each class. """

        if len(features) != 2:
            raise ValueError(f'Features is expected to be a list of 2 column names, got {features}')

        # Selecting non-nan (input, target) data.
        columns = self._df.columns
        x = self._x[:, [columns.get_loc(features[0]), columns.get_loc(features[1])]]
        y = self._y

        # Fitting the classifier into the data.
        num_classes = len(self._class_names)
        clf = self._model.build_classifier(input_size=2, num_classes=num_classes)
        clf.fit(x, y)

        colormap = plt.cm.coolwarm
        _, ax = plt.subplots(constrained_layout=True)
        ax = DecisionBoundaryDisplay.from_estimator(
            clf,
            x,
            grid_resolution=200,
            response_method='predict',
            cmap=colormap,
            alpha=0.8,
            xlabel=features[0],
            ylabel=features[1],
            ax=ax
        ).ax_

        ax.scatter(x[:, 0], x[:, 1], c=y, cmap=colormap, s=20, edgecolors='k')

        # Create legend for each label.
        colours = [colormap(i/(num_classes - 1)) for i in range(num_classes)]
        legend_handles = [mpatches.Patch(color=colours[i], label=name) for i, name in enumerate(self._class_names)]

        ax.legend(handles=legend_handles, title='Class', loc='upper right')
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_title(f'{self._model.model_id} - Decision Boundaries')
        return ax

    def partial_dependence_plot(self, feature: str, target: str) -> Axes:
        """ Generates the partial dependence plot for the specified (feature, target) pair. """

        clf = self._model.classifier
        target_index = self._class_names.index(target)
        _, ax = plt.subplots(constrained_layout=True)
        ax = shap.partial_dependence_plot(
            ind=feature,
            model=lambda df: clf.predict_proba(df)[:, target_index],
            data=self._df,
            ice=False,
            model_expected_value=True,
            feature_expected_value=True,
            show=False,
            ax=ax
        )
        return ax

    def instance_waterfall_plot(self, match_index: int, target: str) -> Axes:
        """ Generates the instance waterfall to explain a specified match. """

        if self._shap_values is None:
            raise RuntimeError('Shap values are required. Call "compute_shap_values" before calling this method.')

        target_index = self._class_names.index(target)
        _, ax = plt.subplots(constrained_layout=True)
        ax = shap.plots.waterfall(
            shap_values=self._shap_values[match_index, :, target_index],
            max_display=self._max_features,
            show=False
        )
        return ax

    def shap_bar_plot(self, target: str, clustering: bool = False) -> Optional[Axes]:
        """ Generates shap importance values (impact on model performance).
            Clustering can help deal with correlated features by clustering correlated features.
        """

        if self._shap_values is None:
            return None

        if clustering and self._cluster is None:
            self._cluster = shap.utils.hclust(self._df, self._y)

        target_index = self._class_names.index(target)
        _, ax = plt.subplots(constrained_layout=True)
        ax = shap.plots.bar(
            self._shap_values[..., target_index],
            max_display=self._max_features,
            clustering=self._cluster,
            show=False,
            ax=ax
        )
        return ax
