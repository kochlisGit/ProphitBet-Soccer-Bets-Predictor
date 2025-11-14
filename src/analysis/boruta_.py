import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional
from boruta import BorutaPy
from matplotlib.axes import Axes
from sklearn.ensemble import RandomForestClassifier
from src.analysis.analyzer import FeatureAnalyzer
from src.preprocessing.dataset import DatasetPreprocessor
from src.preprocessing.utils.normalization import NormalizerType
from src.preprocessing.utils.target import TargetType


class BorutaAnalyzer(FeatureAnalyzer):
    """ Boruta Analyzer that eliminates the least important features using Boruta algorithm. """

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

        # Compute Boruta rankings.
        feat_df = self._compute_boruta_rankings(df=df, target_type=target_type).iloc[: self._max_features]

        # Generate importance bar plot.
        _, ax = plt.subplots(constrained_layout=True)
        ax = sns.barplot(x=feat_df['Ranking'], y=feat_df['Features'], palette=colormap, edgecolor='black', ax=ax)
        ax.tick_params(axis='both', labelsize='small')
        ax.set_title(f'Boruta Feature Rankings for {target_type.name}')
        return ax

    def _compute_boruta_rankings(self, df: pd.DataFrame, target_type: TargetType) -> pd.DataFrame:
        """ Fits a Random Forest Classifier in the dataset and returns the Gini Impurity scores. """

        # Construct & Normalize inputs.
        x, y, _ = self._preprocessor.preprocess_dataset(
            df=df,
            target_type=target_type,
            normalizer=NormalizerType.STANDARD,
            sampler=None
        )

        # Apply Boruta elimination algorithm.
        model = RandomForestClassifier(
            class_weight='balanced',
            random_state=0,
            n_jobs=-1
        )
        feat_selector = BorutaPy(estimator=model, n_estimators='auto', random_state=0)
        feat_selector.fit(x, y)

        feat_df = pd.DataFrame({
            'Features': self._trainable_features,
            'Ranking': feat_selector.ranking_.astype(dtype=int)
        })
        feat_df = feat_df.sort_values(by='Ranking', ascending=True, ignore_index=True)
        return feat_df
