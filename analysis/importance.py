import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from analysis.analyzer import FeatureAnalyzer
from models.tasks import ClassificationTask
from preprocessing.dataset import DatasetPreprocessor


class ImportanceAnalyzer(FeatureAnalyzer):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df=df, preprocess=False)

        self._dataset_preprocessor = DatasetPreprocessor()
        self._x = None
        self._rf_importance_scores = {}

    def plot(self, ax, task: ClassificationTask = ClassificationTask.Result, **kwargs):
        if task not in self._rf_importance_scores:
            if self._x is None:
                self._x = self._dataset_preprocessor.preprocess_inputs(df=self.input_df, return_dataframe=True)

            y = self._dataset_preprocessor.preprocess_targets(df=self.input_df, task=task)
            clf = RandomForestClassifier(random_state=0, n_jobs=-1)
            clf.fit(self._x, y)
            self._rf_importance_scores[task] = pd.DataFrame({
                'Score': clf.feature_importances_,
                'Feature': self._x.columns.tolist()
            }).sort_values(by='Score', ascending=False, ignore_index=True)

        self._rf_importance_scores[task].plot.bar(x='Feature', y='Score', ax=ax)
