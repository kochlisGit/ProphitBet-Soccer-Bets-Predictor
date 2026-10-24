import numpy as np
import pandas as pd
from typing import Dict
from src.core.analysis.analyzers.analyzer import Analyzer


class DatasetAnalyzer(Analyzer):
    """ Dataset analysis class.
        Extracts basic dataset statistics, such as Number of seasons, teams, missing, duplicates, etc.
    """

    def __init__(self):
        super().__init__(name='Dataset')

    def analyze(self, df: pd.DataFrame, metadata: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        size = df.size

        # Computing dataset's statistics.
        num_observations, num_variables = df.shape
        missing = df.isna()
        num_missing_rows = missing.any(axis=1).sum()
        num_missing_values = missing.sum(axis=None)
        num_duplicates = df.duplicated().sum()
        memory_mb = int(df.memory_usage(index=True, deep=True).sum())/(1024**2)
        report = {
            'Seasons': metadata['Season'].nunique(),
            'Teams': np.unique(metadata[['Home', 'Away']].values.ravel()).shape[0],
            'Avg. Home Wins': metadata['Result'].eq('H').mean().round(decimals=self._decimals),
            'Avg. Away Wins': metadata['Result'].eq('A').mean().round(decimals=self._decimals),
            'Avg. Home Goals': metadata['HG'].mean().round(decimals=self._decimals),
            'Avg. Away Goals': metadata['AG'].mean().round(decimals=self._decimals),
            'Observations': num_observations,
            'Variables': num_variables,
            'Missing Values': num_missing_values,
            'Missing Values (%)': round(num_missing_values/size, self._decimals),
            'Duplicates': num_duplicates,
            'Trainable Observations': num_observations - num_missing_rows - num_duplicates,
            'Memory Size (MB)': round(memory_mb, self._decimals),
            'Memory Size / Record (MB)': round(memory_mb/size, self._decimals)
        }
        return {'overview': pd.DataFrame(data={'Value': pd.Series(report, dtype=object)})}
