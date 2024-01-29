import pandas as pd
from preprocessing.dataset import DatasetPreprocessor
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from analysis.analyzer import FeatureAnalyzer


class VarianceAnalyzer(FeatureAnalyzer):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df=df, preprocess=False)

        self._dataset_preprocessor = DatasetPreprocessor()
        self._variance_df = None

    def plot(self, ax, **kwargs):
        if self._variance_df is None:
            inputs = self._dataset_preprocessor.preprocess_inputs(df=self.input_df.dropna(), return_dataframe=True)
            x_scaled, _ = self._dataset_preprocessor.normalize_inputs(x=inputs, normalizer=MinMaxScaler(), fit=True)

            feature_selector = VarianceThreshold()
            feature_selector.fit_transform(x_scaled)
            self._variance_df = pd.DataFrame({
                'Variance': feature_selector.variances_,
                'Feature': inputs.columns.tolist()
            }).sort_values(by='Variance', ascending=False, ignore_index=True)

        self._variance_df.plot.bar(x='Feature', y='Variance', ax=ax)
