import numpy as np
import pandas as pd
from typing import Any, Dict
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from src.core.analysis.analyzers.analyzer import Analyzer
from src.core.preprocessing.utils.targets import TargetType


class FeatureSelectionAnalyzer(Analyzer):
    """ Feature selection analysis class.
        Ranks numerical features based on their importance using normalized variance and boruta algorithm.
    """

    def __init__(self):
        super().__init__(name='Selection')

    def _variance_scores(self, x: pd.DataFrame, feature_names: pd.DataFrame) -> pd.DataFrame:
        """ Computed the variance score per feature.

                Args
                ----
                :param x: (np.ndarray) inputs.
                :param feature_names: (List[str]) list of feature names.
             """

        # Analyze variance.
        feature_selector = VarianceThreshold()
        feature_selector.fit_transform(x)

        # Generate variances bar plot with descending order.
        variance_df = pd.DataFrame(data={
            'Feature': feature_names, 'Variance': feature_selector.variances_
        }).sort_values(by='Variance', ascending=False, ignore_index=True).round(decimals=self._decimals)
        return variance_df

    def _boruta_rankings(self, x: np.ndarray, y: np.ndarray, feature_names: pd.Series) -> pd.DataFrame:
        """ Uses a random forest classifier to compute boruta rankings.

            Args
            ----
            :param x: (np.ndarray) inputs.
            :param y: (np.ndarray) targets.
            :param feature_names: (List[str]) list of feature names.
        """

        # Initializing & training random forest classifier using boruta method.
        model = RandomForestClassifier(class_weight='balanced', random_state=0, n_jobs=-1)
        feat_selector = BorutaPy(estimator=model, n_estimators='auto', random_state=0)
        feat_selector.fit(x, y)
        rankings_df = pd.DataFrame(data={
            'Feature': feature_names, 'Rank': feat_selector.ranking_
        }).sort_values(by='Rank', ascending=True, ignore_index=True)
        return rankings_df

    def analyze(self, df: pd.DataFrame, metadata: pd.DataFrame) -> Dict[str, Any]:
        # Removing samples (rows) with nan values.
        valid_ids = df.notna().all(axis=1)
        valid_metadata = metadata[valid_ids]
        columns = df.columns

        x = df.loc[valid_ids].to_numpy(dtype=np.float32)
        x_scaled = MinMaxScaler().fit_transform(x)

        selection_dict = {
            'normalized_variance': self._variance_scores(x=x_scaled, feature_names=columns),
            'boruta': {}
        }
        for target in TargetType:
            y = valid_metadata[target.value].map(target.get_mapper()).to_numpy(dtype=np.int32)
            selection_dict['boruta'][target.value] = self._boruta_rankings(x=x, y=y, feature_names=columns)
        return selection_dict
