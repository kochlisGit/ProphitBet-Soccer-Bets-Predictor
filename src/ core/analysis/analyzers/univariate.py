import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
from scipy.stats import gaussian_kde
from src.core.analysis.analyzers.analyzer import Analyzer
from src.core.preprocessing.utils.targets import TargetType


class UnivariateAnalyzer(Analyzer):
    """ Univariate analysis class. Analyzes distributions, imbalances and missing/unique counts of each variable. """

    def __init__(self):
        super().__init__(name='Univariate')

        self._min_discrete_unique_vals = 10
        self._entropy_threshold = 0.65
        self._dominant_prob_threshold = 0.50

    def _compute_descriptive_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes descriptive statistics (mean, var, std, etc.) for each variable.
            Returns a dict {column: {statistic: value}}

            Example
            -------
            {'1': {'Unique': ..., 'Missing': ...}, ...}
        """

        missing = df.isna().sum()
        minimum = df.min()
        maximum = df.max()
        quantiles = df.quantile([0.25, 0.75])
        q1 = quantiles.loc[0.25]
        q3 = quantiles.loc[0.75]

        statistics = pd.DataFrame(data={
            'Unique': df.nunique(),
            'Missing': missing,
            'Missing (%)': missing.div(df.shape[0])*100,
            'Zeros': df.eq(0).sum(),
            'Positives': df.gt(0).sum(),
            'Negatives': df.lt(0).sum(),
            'Mean': df.mean(),
            'Median': df.median(),
            'Mode': df.mode().iloc[0].reindex(df.columns),
            'Variance': df.var(),
            'Standard Deviation': df.std(),
            'Max': maximum,
            'Min': minimum,
            'Range': maximum.sub(minimum),
            'Q1 (0.25)': q1,
            'Q3 (0.75)': q3,
            'Interquartile Range': q3.sub(q1),
            'Skew Coeff.': df.skew(),
            'Kurtosis Coeff.': df.kurt()
        })
        statistics = statistics.round(decimals=self._decimals)
        return statistics

    def _find_variable_issues(
            self,
            statistics: pd.DataFrame
    ) -> Dict[str, Tuple[str, List[Tuple[str, float]]]]:
        """ Identify potentially problematic variables from descriptive statistics. Issues are returned as a
            dict [Issue: (Description, List of (Variable, Trigger Value))] from most important to least important.

            Args
            ----------
            :param statistics : (pd.DataFrame) Output of ``_compute_descriptive_statistics``.
        """

        issues = {}
        missing = statistics['Missing (%)']
        unique = statistics['Unique']
        variance = statistics['Variance']
        iqr = statistics['Interquartile Range']
        skew = statistics['Skew Coeff.'].abs()
        kurtosis = statistics['Kurtosis Coeff.']

        # 1. Unusable or almost unusable variables, due to high missing values (%).
        affected = [(column, float(missing.loc[column])) for column in statistics.index[missing > 20]]
        if affected:
            issues['High Missing'] = ('might be unusable, due to many missing values.', affected)

        # 2. Variables with little or no spread.
        low_spread = (
            ((variance == 0) & (unique > 1)) |
            ((iqr == 0) & (unique > 1) & (variance > 0))
        )
        affected = [
            (column, float(variance.loc[column] if variance.loc[column] == 0 else iqr.loc[column]))
            for column in statistics.index[low_spread]
        ]
        if affected:
            issues['Low Variance'] = ('might be irrelevant, due to low variance.', affected)

        # 3. Excess skewness: normal distribution ~= 0.
        affected = [(column, float(skew.loc[column])) for column in statistics.index[skew >= 2]]
        if affected:
            issues['Excess Skewness'] = (
                'might result in lower quality predictions, due to excess skewness.', affected
            )

        # 4. Excess kurtosis: normal distribution ~= 0.
        affected = [
            (column, float(kurtosis.loc[column]))
            for column in statistics.index[(kurtosis >= 3) & (kurtosis < 7)]
        ]
        if affected:
            issues['Excess Kurtosis'] = (
                'might result in lower quality predictions, due to excess kurtosis.', affected
            )

        # 5. Mean-Median Differences.
        valid_iqr = iqr.replace(0, pd.NA)
        mean_median_difference = statistics['Mean'].sub(statistics['Median']).abs() / valid_iqr
        affected = [
            (column, float(mean_median_difference.loc[column]))
            for column in statistics.index[mean_median_difference >= 1]
        ]
        if affected:
            issues['Excess Outliers'] = (
                'might result in lower quality predictions, due to excess number of outliers, which might be ignored',
                affected
            )

        return issues

    def _compute_distributions(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """ Compute discrete value counts (Bar) or continuous Histogram + KDE data per variable.

            Args
            ----
            :param df: (pd.DataFrame): League's dataframe.
        """

        distributions = {}
        for column in df.columns:
            valid_df = df[column].dropna()
            n_unique = valid_df.nunique()

            # If few observations, then barplot else computes histogram.
            if n_unique < self._min_discrete_unique_vals:
                distributions[column] = {'Type': 'Discrete', 'Data': valid_df.value_counts()}
            else:
                values = valid_df.to_numpy(dtype=np.float32)
                counts, edges = np.histogram(values, bins=10)
                widths = np.diff(edges)
                centers = edges[:-1] + widths/2.0

                # Computing KDE: returns probability density. Scaling it to histogram counts so both can be overlaid.
                kde = gaussian_kde(values)
                kde_x = np.linspace(edges[0], edges[-1],200)
                kde_y = kde(kde_x) * values.shape[0] * widths.mean()
                distributions[column] = {
                    'Type': 'Continuous',
                    'Data': {
                        'counts': counts,
                        'bin_edges': edges,
                        'bin_widths': widths,
                        'bin_centers': centers,
                        'kde_x': kde_x,
                        'kde_y': kde_y
                    }
                }
        return distributions

    def _compute_imbalances(self, df: pd.DataFrame) -> List[Tuple[str, float]]:
        """ Return imbalanced variables together with their normalized entropy scores. """

        def normalized_entropy(counts: pd.Series) -> Tuple[float, float]:
            """ Returns the normalized entropy score and dominant probability given the value counts of a variable.

                Args
                ----
                :param counts: (pd.Series) counts of given variable.
            """

            if counts.size == 0:
                return 0.0, 0.0
            if counts.size == 1:
                return 0.0, 1.0

            # Computing probabilities and dominant probability. Non-zeros
            probabilities = counts/counts.sum()
            dominant_prob = probabilities.max()

            # Filtering zero probabilities (for histograms only, as bar plots include at least a single count).
            nonzero_probabilities = probabilities[probabilities > 0]
            entropy = -np.sum(nonzero_probabilities * np.log(nonzero_probabilities))

            # Computing normalized entropy score, dominant probability.
            norm_entropy = entropy / np.log(counts.shape[0])
            return norm_entropy, dominant_prob

        imbalanced_variables = []
        for column in df.columns:
            counts = df[column].value_counts()
            norm_entropy, dominant_prob = normalized_entropy(counts=counts)
            if norm_entropy < self._entropy_threshold or dominant_prob > self._dominant_prob_threshold:
                imbalanced_variables.append((column, round(float(norm_entropy), self._decimals)))

        return imbalanced_variables

    def analyze(self, df: pd.DataFrame, metadata: pd.DataFrame) -> Dict[str, Any]:
        descriptive_statistics = self._compute_descriptive_statistics(df=df)
        variable_issues = self._find_variable_issues(statistics=descriptive_statistics)

        # Including numerical metadata in the analysis.
        meta_cols = ['1X', '12', 'X2', 'HG', 'AG'] + [target.value for target in TargetType]
        numerical_df = pd.concat([df, metadata[meta_cols]], axis=1)

        distributions = self._compute_distributions(df=numerical_df)
        imbalances = self._compute_imbalances(df=numerical_df)
        return {
            'descriptive_statistics': descriptive_statistics,
            'distributions': distributions,
            'issues': variable_issues,
            'imbalanced_variables': imbalances
        }
