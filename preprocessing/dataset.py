import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import RandomOverSampler, SVMSMOTE
from imblearn.combine import SMOTEENN
from models.tasks import ClassificationTask


class DatasetPreprocessor:
    def __init__(self):
        self._columns_to_drop = ['Date', 'Season', 'Home Team', 'Away Team', 'HG', 'AG', 'Result']
        self._target_fn = {
            ClassificationTask.Result: lambda df: df['Result'].replace({'H': 0, 'D': 1, 'A': 2}).to_numpy(dtype=np.int32),
            ClassificationTask.Over: lambda df: (df['HG'] + df['AG'] > 2).to_numpy(dtype=np.int32)
        }

    @staticmethod
    def _get_normalizer(normalizer_str: str) -> TransformerMixin or None:
        if normalizer_str == 'None':
            return None
        elif normalizer_str == 'Min-Max':
            return MinMaxScaler()
        elif normalizer_str == 'Max-Abs':
            return MaxAbsScaler()
        elif normalizer_str == 'Standard':
            return StandardScaler()
        elif normalizer_str == 'Robust':
            return RobustScaler()
        else:
            raise NotImplementedError(f'Undefined normalizer: "{normalizer_str}"')

    @staticmethod
    def _get_sampler(sampler_str: str) -> TransformerMixin or None:
        if sampler_str == 'None':
            return None
        elif sampler_str == 'Random-UnderSampling':
            return RandomUnderSampler(random_state=0)
        elif sampler_str == 'Near-Miss':
            return NearMiss(version=3)
        elif sampler_str == 'Random-OverSampling':
            return RandomOverSampler(random_state=0)
        elif sampler_str == 'SVM-SMOTE':
            return SVMSMOTE(random_state=0)
        elif sampler_str == 'SMOTE-NN':
            return SMOTEENN(random_state=0)
        else:
            raise NotImplementedError(f'Undefined sampler: "{sampler_str}"')

    def preprocess_inputs(self, df: pd.DataFrame, return_dataframe: bool = False) -> np.ndarray or pd.DataFrame:
        x = df.drop(columns=self._columns_to_drop)
        return x if return_dataframe else x.to_numpy(dtype=np.float64)

    def preprocess_targets(self, df: pd.DataFrame, task: ClassificationTask) -> np.ndarray:
        return self._target_fn[task](df=df)

    @staticmethod
    def normalize_inputs(
            x: np.ndarray,
            normalizer: TransformerMixin or None,
            fit: bool
    ) -> (np.ndarray, TransformerMixin or None):
        if normalizer is None:
            return x, None
        elif fit:
            x = normalizer.fit_transform(x)
            return x, normalizer
        else:
            x = normalizer.transform(x)
            return x, normalizer

    @staticmethod
    def sample_inputs(
            x: np.ndarray,
            y: np.ndarray or None,
            sampler: TransformerMixin or None
    ) -> (np.ndarray, np.ndarray or None, TransformerMixin or None):
        if sampler is None:
            return x, y, None
        else:
            x, y = sampler.fit_resample(x, y)
            return x, y, sampler

    def preprocess_dataset(
            self,
            df: pd.DataFrame,
            task: ClassificationTask,
            fit_normalizer: bool,
            normalizer: TransformerMixin or str or None,
            sampler: TransformerMixin or str or None
    ) -> (np.ndarray, np.ndarray, TransformerMixin or None, TransformerMixin or None):
        assert not df.isna().any().any(), 'Cannot preprocess dataframe with nan values'

        if isinstance(normalizer, str):
            normalizer = self._get_normalizer(normalizer_str=normalizer)
        if isinstance(sampler, str):
            sampler = self._get_sampler(sampler_str=sampler)

        x = self.preprocess_inputs(df=df, return_dataframe=False)
        y = self.preprocess_targets(df=df, task=task)
        x, normalizer = self.normalize_inputs(x=x, normalizer=normalizer, fit=fit_normalizer)
        x, y, sampler = self.sample_inputs(x=x, y=y, sampler=sampler)
        return x, y, normalizer, sampler

    @staticmethod
    def construct_input(
            matches_df: pd.DataFrame,
            home_team: str,
            away_team: str,
            odd_1: float or None,
            odd_x: float or None,
            odd_2: float or None
    ) -> np.ndarray:
        home_matches = matches_df[matches_df['Home Team'] == home_team]
        home_columns = home_matches[[col for col in matches_df.columns if col[0] == 'H' and col != 'HG' and col != 'Home Team']].head(1).values
        away_matches = matches_df[matches_df['Away Team'] == away_team]
        away_columns = away_matches[[col for col in matches_df.columns if col[0] == 'A' and col != 'AG' and col != 'Away Team']].head(1).values
        odds_list = [odd for odd, col in zip([odd_1, odd_x, odd_2], ['1', 'X', '2']) if col in matches_df.columns]
        odd_columns = np.array([odds_list], dtype=home_columns.dtype)
        return np.hstack((odd_columns, home_columns, away_columns))
