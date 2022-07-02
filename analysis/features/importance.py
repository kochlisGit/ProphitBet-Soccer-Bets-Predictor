from analysis.features.analyzer import FeatureAnalyzer
from xgboost import XGBRFClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.feature_selection import RFE
import pandas as pd


class ImportanceAnalyzer(FeatureAnalyzer):
    def __init__(self, results_and_stats: pd.DataFrame):
        super().__init__(results_and_stats)
        self._clf_weight_model = None
        self._rfe_model = None
        self._variance_model = None
        self._best_model = None

    def plot_feature_classification_weights(self, color, ax):
        if self._clf_weight_model is None:
            self._clf_weight_model = XGBRFClassifier(n_jobs=-1)
            self._clf_weight_model.fit(self.inputs, self.targets)

        weights = self._clf_weight_model.get_booster().get_score(importance_type='weight')
        self._plot_bar(data=list(weights.values()), labels=list(weights.keys()), color=color, ax=ax)

    def plot_feature_elimination_importance(self, color, ax):
        if self._rfe_model is None:
            self._rfe_model = RFE(
                estimator=XGBRFClassifier(n_jobs=-1),
                step=1
            )
            self._rfe_model.fit(self._inputs, self.targets)

        self._plot_bar(data=self._rfe_model.ranking_, labels=self._labels, color=color, ax=ax)

    def plot_feature_variances(self, color, ax):
        if self._variance_model is None:
            self._variance_model = VarianceThreshold()
            self._variance_model.fit(self._inputs)

        self._plot_bar(data=self._variance_model.variances_, labels=self._labels, color=color, ax=ax)

    def plot_univariate_test_importance(self, color, ax):
        if self._best_model is None:
            self._best_model = SelectKBest(score_func=f_classif, k='all')
            self._best_model.fit(self._inputs, self.targets)

        self._plot_bar(data=self._best_model.scores_, labels=self._labels, color=color, ax=ax)
