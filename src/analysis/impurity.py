import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional
from matplotlib.axes import Axes
from sklearn.ensemble import RandomForestClassifier
from src.analysis.analyzer import FeatureAnalyzer
from src.preprocessing.dataset import DatasetPreprocessor
from src.preprocessing.utils.normalization import NormalizerType
from src.preprocessing.utils.target import TargetType


class GiniImpurityAnalyzer(FeatureAnalyzer):
    """ Gini-Impurity-based analyzer, which analyzes the feature importance via a fitted Random Forest. """

    def __init__(self, df: pd.DataFrame):
        super().__init__(df=df)

        self._preprocessor = DatasetPreprocessor()
        self._max_features = 20

    def _generate_plot(
            self,
            df: pd.DataFrame,
            colormap: Optional[str] = None,
            target_type: TargetType = None
    ) -> Axes:
        """
            Applies Standard scaling to data, computes impurity-based feature importance scores using
            a Random Forest Classifier and generates the corresponding bar plot.
        """

        # Compute Gini impurity-based feature importance scores.
        scores = self._compute_impurity_scores(df=df, target_type=target_type)

        # Sort importance scores.
        scores_df = pd.DataFrame({'Features': self._trainable_features, 'Gini Impurity': scores})
        scores_df = scores_df.sort_values(by='Gini Impurity', ascending=False, ignore_index=True).iloc[: self._max_features]

        # Generate importance bar plot.
        _, ax = plt.subplots(constrained_layout=True)
        ax = sns.barplot(x=scores_df['Gini Impurity'], y=scores_df['Features'], palette=colormap, ax=ax)
        ax.set_ylabel(None)
        ax.tick_params(axis='both', labelsize='small')
        ax.set_title(f'Gini Impurity Bar Plot for {target_type.name}')
        return ax

    def _compute_impurity_scores(self, df: pd.DataFrame, target_type: TargetType) -> np.ndarray:
        """ Fits a Random Forest Classifier in the dataset and returns the Gini Impurity scores. """

        # Construct & Normalize inputs.
        x, y, _ = self._preprocessor.preprocess_dataset(
            df=df,
            target_type=target_type,
            normalizer=NormalizerType.STANDARD,
            sampler=None
        )

        # Apply Linear Regression with Lasso (L1 Penalty).
        model = RandomForestClassifier(
            class_weight='balanced',
            random_state=0,
            n_jobs=-1
        )
        model.fit(x, y)
        return model.feature_importances_
