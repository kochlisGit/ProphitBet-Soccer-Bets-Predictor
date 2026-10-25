import pandas as pd
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from src.core.analysis.analyzers import analyzers
from src.core.analysis.analyzers.analyzer import Analyzer
from src.core.preprocessing.dataset import DatasetPreprocessor


class ExploratoryDataAnalyzer:
    """ Class for exploratory data analysis.

        Features:
            1. Dataset overview.
            2. Univariate analysis (Distributions, Imbalances, Missing, Counts).
            3. Bivariate analysis (Scatter plots, Correlation inputs-inputs, Correlation inputs-targets).
            4. Correlation analysis.
            5. Feature Selection analysis (Variance, Boruta).
            6. Feature Importance analysis (Coefficient, Gini Impurity).
            7. Rule extraction.
            8. Team performance analysis.
    """

    def __init__(self):
        self._analyzers = {analyzer.name: analyzer for analyzer in analyzers}

    @property
    def analyzers(self) -> Dict[str, Analyzer]:
        return self._analyzers

    def _select_analyzers(self, eda_analyzers: Optional[List[str]] = None) -> Dict[str, Analyzer]:
        """ Helper function that fetches the desired analyzers by name. """

        if eda_analyzers is None:
            return self._analyzers

        return {name: self._analyzers[name] for name in eda_analyzers}

    def exploratory_data_analysis(self, df: pd.DataFrame, eda_analyzers: Optional[List[str]] = None) -> Dict[str, Any]:
        """ Applies EDA and returns multiple reports for each season and for all seasons.

            Args
            ----
            :param df: (pd.DataFrame) league's computed data and statistics.
            :param eda_analyzers: (List[str | Analyzer] | None) league's eda analyzers.
        """

        # Selecting analyzers to apply EDA.
        selected_analyzers = self._select_analyzers(eda_analyzers=eda_analyzers)

        if not selected_analyzers:
            return {}

        # Preprocessing dataframe: Splitting into Input dataframe and Metadata.
        dataset_preprocessor = DatasetPreprocessor()
        eda_df = df.drop(columns=dataset_preprocessor.metadata_columns)
        metadata = df[dataset_preprocessor.metadata_columns]

        # Generating the EDA report.
        eda_report = {}
        for name, analyzer in tqdm(selected_analyzers.items(), desc='Data Analysis Report'):
            report = analyzer.analyze(df=eda_df, metadata=metadata)
            eda_report[name] = report
        return eda_report
