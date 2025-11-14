import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
from matplotlib.axes import Axes
from sklearn.tree import DecisionTreeClassifier, plot_tree
from src.analysis.analyzer import FeatureAnalyzer
from src.preprocessing.dataset import DatasetPreprocessor
from src.preprocessing.utils.target import TargetType


class RuleExtractorAnalyzer(FeatureAnalyzer):
    """ Rule extractor analyzer using a decision tree. """

    def __init__(self, df: pd.DataFrame):
        super().__init__(df=df)

        self._preprocessor = DatasetPreprocessor()

    def _generate_plot(self, df: pd.DataFrame, colormap: Optional[str] = None, target_type: TargetType = None, max_depth: int = 3) -> Axes:
        """ Fits a Decision Tree in the data and extracts classification rules based on the requested maximum depth. """

        # Fit tree.
        tree = self._fit_tree(df=df, target_type=target_type, max_depth=max_depth)

        if target_type == TargetType.RESULT:
            classes = ['H', 'D', 'A']
        elif target_type == TargetType.OVER_UNDER:
            classes = ['U', 'O']
        else:
            raise TypeError(f'Undefiend target type: "{target_type.name}"')

        _, ax = plt.subplots(constrained_layout=True)
        ax = plot_tree(
            decision_tree=tree,
            max_depth=max_depth,
            feature_names=self._trainable_features,
            class_names=classes,
            proportion=True,
            rounded=True,
            fontsize=6,
            ax=ax
        )
        return ax

    def _fit_tree(self, df: pd.DataFrame, target_type: TargetType, max_depth: int) -> DecisionTreeClassifier:
        """ Fits a Decision Tree into the data. """

        # Construct & Normalize inputs.
        x, y, _ = self._preprocessor.preprocess_dataset(
            df=df,
            target_type=target_type,
            normalizer=None,
            sampler=None
        )

        model = DecisionTreeClassifier(max_depth=max_depth, class_weight='balanced')
        model.fit(x, y)
        return model
