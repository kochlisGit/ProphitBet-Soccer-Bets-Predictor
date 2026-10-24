import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
from itertools import combinations
from dcor import distance_correlation
from src.core.analysis.analyzers.analyzer import Analyzer


class BivariateAnalyzer(Analyzer):
    """ Bivariate analysis class. Analyzes movements, correlations between variables and interactions between features. """

    def __init__(self):
        super().__init__(name='Bivariate')

        self._corr_thresholds = {'Pearson': 0.70, 'Spearman': 0.70, 'Kendall': 0.5, 'Distance': 0.70, 'Eta2': 0.1}

    def _correlated_pairs(self, correlations: Dict[str, pd.DataFrame]) -> List[Tuple[str, str, float]]:
        """ Extract correlated feature pairs from precomputed correlation matrices.
            Returns each correlated feature pair with its strongest qualifying correlation value.

            Args
            ----
            :param correlations: (Dict[str, pd.DataFrame]) correlation matrices for each method.
        """

        correlated_pairs = {}
        for method, corr in correlations.items():
            threshold = self._corr_thresholds[method]

            # Feature correlations require only the upper/lower triangle.
            mask = np.triu(corr.values > threshold, k=1)
            rows, columns = np.where(mask)

            # Extracting correlated pairs. If multiple methods identify the same pair, retain the strongest value.
            for row, column in zip(rows, columns):
                pair = (corr.index[row], corr.columns[column])
                value = float(corr.iloc[row, column])
                if pair not in correlated_pairs or value > correlated_pairs[pair]:
                    correlated_pairs[pair] = value

        return [(*pair, value) for pair, value in sorted(correlated_pairs.items())]

    def analyze(self, df: pd.DataFrame, metadata: pd.DataFrame) -> Dict[str, Any]:
        pearson = df.corr(method='pearson').round(decimals=self._decimals)
        spearman = df.corr(method='spearman').round(decimals=self._decimals)
        kendall = df.corr(method='kendall').round(decimals=self._decimals)
        distance = df.corr(method=distance_correlation).round(decimals=self._decimals)
        correlated_pairs = self._correlated_pairs(correlations={
            'Pearson': pearson, 'Spearman': spearman, 'Kendall': kendall, 'Distance': distance
        })
        interactions = {f'{c1} - {c2}': df[[c1, c2]] for c1, c2 in combinations(df.columns, 2)}
        return {
            'feature_correlations': {
                'Pearson': pearson,
                'Spearman': spearman,
                'Kendall': kendall,
                'Distance': distance
            },
            'correlated_pairs': correlated_pairs,
            'interactions': interactions
        }
