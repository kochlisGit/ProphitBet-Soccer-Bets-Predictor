import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from src.core.analysis.analyzers.analyzer import Analyzer
from src.core.preprocessing.utils.targets import TargetType


class FeatureImportanceAnalyzer(Analyzer):
    """ Feature importance analysis class. Ranks numerical features based on their
        coefficient and impurity importance scores using Logistic Regression and Random Forest Classifiers.
    """

    def __init__(self):
        super().__init__(name='Importance')

    def _linear_coefficients(
            self,
            x: np.ndarray,
            y: np.ndarray,
            feature_names: List[str],
            target_names: List[str]
    ) -> pd.DataFrame:
        """ Uses a logistic regression classifier to compute linear coefficients of the features.

            Args
            ----
            :param x: (np.ndarray) normalized inputs.
            :param y: (np.ndarray) encoded targets.
            :param feature_names: (List[str]) list of feature names.
            :param target_names: (List[str]) list of target names.
        """

        model = LogisticRegressionCV(
            Cs=[1e-2, 1e-1, 5e-1, 1.0, 10],
            l1_ratios=[0.01, 0.1, 0.2, 0.5, 1.0],
            solver='saga',
            scoring='accuracy',
            max_iter=5000,
            random_state=0,
            n_jobs=-1,
            use_legacy_attributes=False
        )
        model.fit(x, y)

        # Return absolute coefficients dataframe: columns: feature names, rows: coefficients. index: target.
        abs_coeffs = np.abs(model.coef_)
        n_classes = len(target_names)

        if n_classes == 2:
            coeff_df = pd.DataFrame(
                data={'Feature': feature_names, 'Coefficient': abs_coeffs[0]}
            ).sort_values(by='Coefficient', ascending=False, ignore_index=True)
        else:
            coeff_df = pd.DataFrame(data=abs_coeffs.transpose(), columns=target_names, index=feature_names)
        return coeff_df.round(decimals=self._decimals)

    def _gini_impurity_scores(self, x: np.ndarray, y: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """ Uses a random forest classifier to compute gini impurity scores of the features.

            Args
            ----
            :param x: (np.ndarray) inputs.
            :param y: (np.ndarray) encoded targets.
            :param feature_names: (List[str) list of feature names.
        """

        model = RandomForestClassifier(class_weight='balanced', random_state=0, n_jobs=-1)
        model.fit(x, y)
        scores_df = pd.DataFrame(
            data={'Features': feature_names, 'Gini Impurity': model.feature_importances_}
        ).sort_values(by='Gini Impurity', ascending=False, ignore_index=True).round(decimals=self._decimals)
        return scores_df


    def analyze(self, df: pd.DataFrame, metadata: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
        # Removing samples (rows) with nan values.
        valid_ids = df.notna().all(axis=1)
        valid_df = df[valid_ids]
        valid_metadata = metadata[valid_ids]
        columns = df.columns

        # Constructing inputs.
        x = valid_df.to_numpy(dtype=np.float32)
        x_scaled = StandardScaler().fit_transform(x)

        # Constructing targets and calculating coefficients for each target type.
        importance_dict = {'linear_coefficients': {}, 'gini_impurity_scores': {}}
        for target in TargetType:
            mapper = target.get_mapper()
            y = valid_metadata[target.value].map(mapper).to_numpy(dtype=np.int32)
            importance_dict['linear_coefficients'][target.value] = self._linear_coefficients(
                x=x_scaled, y=y, feature_names=columns, target_names=list(mapper.keys())
            )
            importance_dict['gini_impurity_scores'][target.value] = self._gini_impurity_scores(
                x=x, y=y, feature_names=columns
            )
        return importance_dict
