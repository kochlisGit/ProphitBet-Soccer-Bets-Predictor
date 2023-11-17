import numpy as np
import seaborn
import pandas as pd
from analysis.features.analyzer import FeatureAnalyzer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.feature_selection import RFE
from xgboost import XGBRFClassifier


class ImportanceAnalyzer(FeatureAnalyzer):
    def __init__(self, matches_df: pd.DataFrame):
        super().__init__(matches_df=matches_df)

        self._class_weights_model = None
        self._rfe_model = None
        self._variance_model = None
        self._best_model = None

    def plot_feature_classification_weights(self, ax):
        if self._class_weights_model is None:
            self._class_weights_model = XGBRFClassifier(random_state=0, n_jobs=-1)
            self._class_weights_model.fit(self.inputs, self.targets)

        weights = self._class_weights_model.get_booster().get_score(
            importance_type="weight"
        )
        self.plot(x=list(weights.values()), y=list(weights.keys()), ax=ax)

    def plot_feature_elimination_importance(self, ax):
        if self._rfe_model is None:
            self._rfe_model = RFE(
                estimator=XGBRFClassifier(random_state=0, n_jobs=-1), step=1
            )
            self._rfe_model.fit(self._inputs, self.targets)

        self.plot(x=self._rfe_model.ranking_, y=self.columns, ax=ax)

    def plot_feature_variances(self, ax):
        if self._variance_model is None:
            self._variance_model = VarianceThreshold()
            self._variance_model.fit(self._inputs)

        self.plot(x=self._variance_model.variances_, y=self.columns, ax=ax)

    def plot_univariate_test_importance(self, ax):
        if self._best_model is None:
            self._best_model = SelectKBest(score_func=f_classif, k="all")
            self._best_model.fit(self._inputs, self.targets)

        self.plot(x=self._best_model.scores_, y=self.columns, ax=ax)

    def plot(self, x: np.ndarray or list, y: np.ndarray or list, ax, **kwargs):
        seaborn.barplot(x=x, y=y, ax=ax)
